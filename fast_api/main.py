from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import re
from typing import List
from pydantic import BaseModel
from joblib import load

app = FastAPI()

# Загрузка модели и OneHotEncoder
model = load('model.joblib')
ohe = pickle.load(open('onehotencoder.pkl', 'rb'))

# Получаем признаки, которые использовались при обучении модели
original_feature_names = ohe.get_feature_names_out()

class DataPreprocessing:
    def __init__(self, data):
        if isinstance(data, dict):
            self.df = pd.DataFrame(data, index=[0])
        else:
            self.df = data

    # Убираем единицы измерения
    def extract_number(self, value):
        match = re.search(r'\d+(?:\.\d+)?', str(value))
        return float(match.group()) if match else np.nan

    def data_cleaning(self):
        change_columns = ['mileage', 'engine', 'max_power']
        for column in change_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].apply(self.extract_number)

        # Заполняем пропуски медианными значениями
        median = pickle.load(open('median.pkl', 'rb'))
        fill_columns = ['mileage', 'engine', 'max_power', 'seats']
        for col in fill_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(median[col])

        # Приведение типов
        if 'engine' in self.df.columns:
            self.df['engine'] = self.df['engine'].astype(int)
        if 'seats' in self.df.columns:
            self.df['seats'] = self.df['seats'].astype(int)
        return self

    # Удаление столбца torque
    def drop_col_torque(self):
        if 'torque' in self.df.columns:
            self.df = self.df.drop(columns=['torque'])
        return self

    # Кодирование категориальных признаков
    def ohe(self):
        categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
        if all(feature in self.df.columns for feature in categorical_features):
            encoded_data = ohe.transform(self.df[categorical_features])
            encoded_df = pd.DataFrame(encoded_data.toarray(), columns=ohe.get_feature_names_out(categorical_features))
            self.df = self.df.join(encoded_df)
            self.df.drop(categorical_features, axis=1, inplace=True)

        # Удаляем целевой столбец selling_price
        if 'selling_price' in self.df.columns:
            self.df.drop(columns=['selling_price'], inplace=True)

        # Оставляем только числовые признаки
        self.df = self.df.select_dtypes(include=['int', 'float', 'bool']).copy()
        return self.df


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


# Предсказание для одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> str:
    data = item.dict()
    formatting = DataPreprocessing(data)
    formatting_data = formatting.data_cleaning().drop_col_torque().ohe()

    predict = model.predict(formatting_data)
    return f'Predicted car price: {predict[0]:.2f}'


@app.post("/predict_items", response_class=StreamingResponse)
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    df_test = pd.read_csv(BytesIO(content))

    required_columns = ['mileage', 'engine', 'max_power', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    missing_columns = [col for col in required_columns if col not in df_test.columns]
    if missing_columns:
        return {"error": f"The following required columns are missing: {', '.join(missing_columns)}"}

    if 'selling_price' in df_test.columns:
        df_test.drop(columns=['selling_price'], inplace=True)

    formatting = DataPreprocessing(df_test)
    formatting_data = formatting.data_cleaning().drop_col_torque().ohe()

    predict = model.predict(formatting_data)
    df_test['predicted_price'] = predict

    output_stream = BytesIO()
    df_test.to_csv(output_stream, index=False)
    output_stream.seek(0)

    response = StreamingResponse(output_stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv"
    return response