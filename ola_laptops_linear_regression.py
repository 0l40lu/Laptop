import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('laptopPrice.csv')
df = data[['brand',
           'processor_brand',
           'processor_name',
           'processor_gnrtn',
           'ram_gb',
           'ram_type',
           'ssd',
           'hdd',
           'os',
           'os_bit',
           'graphic_card_gb',
           'weight',
           'warranty',
           'Touchscreen',
           'msoffice',
           'Price']].dropna()

brand_encoder = LabelEncoder()
df['brand'] = brand_encoder.fit_transform(df['brand'])

processor_brand_encoder = LabelEncoder()
df['processor_brand'] = processor_brand_encoder.fit_transform(df['processor_brand'])

processor_name_encoder = LabelEncoder()
df['processor_name'] = processor_name_encoder.fit_transform(df['processor_name'])

processor_gnrtn_encoder = LabelEncoder()
df['processor_gnrtn'] = processor_gnrtn_encoder.fit_transform(df['processor_gnrtn'])

ram_gb_encoder = LabelEncoder()
df['ram_gb'] = ram_gb_encoder.fit_transform(df['ram_gb'])

ram_type_encoder = LabelEncoder()
df['ram_type'] = ram_type_encoder.fit_transform(df['ram_type'])

ssd_encoder = LabelEncoder()
df['ssd'] = ssd_encoder.fit_transform(df['ssd'])

hdd_encoder = LabelEncoder()
df['hdd'] = hdd_encoder.fit_transform(df['hdd'])

os_encoder = LabelEncoder()
df['os'] = os_encoder.fit_transform(df['os'])

os_bit_encoder = LabelEncoder()
df['os_bit'] = os_bit_encoder.fit_transform(df['os_bit'])

graphic_card_gb_encoder = LabelEncoder()
df['graphic_card_gb'] = graphic_card_gb_encoder.fit_transform(df['graphic_card_gb'])

weight_encoder = LabelEncoder()
df['weight'] = weight_encoder.fit_transform(df['weight'])

warranty_encoder = LabelEncoder()
df['warranty'] = warranty_encoder.fit_transform(df['warranty'])

touchscreen_encoder = LabelEncoder()
df['Touchscreen'] = touchscreen_encoder.fit_transform(df['Touchscreen'])

msoffice_encoder = LabelEncoder()
df['msoffice'] = msoffice_encoder.fit_transform(df['msoffice'])

x = df[['brand',
        'processor_brand',
        'processor_name',
        'processor_gnrtn',
        'ram_gb',
        'ram_type',
        'ssd',
        'hdd',
        'os',
        'os_bit',
        'graphic_card_gb',
        'weight',
        'warranty',
        'Touchscreen',
        'msoffice']]
y = df['Price']

feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(feature_train, target_train)

st.header('Laptop Price Range Created by Ola')
brand = st.sidebar.selectbox('Brand', brand_encoder.classes_)
processor_brand = st.sidebar.selectbox('Processor Brand', processor_brand_encoder.classes_)
processor_name = st.sidebar.selectbox('Processor Name', processor_name_encoder.classes_)
processor_gnrtn = st.sidebar.selectbox('Process Generation', processor_gnrtn_encoder.classes_)
ram_gb = st.sidebar.selectbox('Ram GB', ram_gb_encoder.classes_)
ram_type = st.sidebar.selectbox('Ram Type', ram_type_encoder.classes_)
ssd = st.sidebar.selectbox('SSD', ssd_encoder.classes_)
hdd = st.sidebar.selectbox('HDD', hdd_encoder.classes_)
os = st.sidebar.selectbox('OS', os_encoder.classes_)
os_bit = st.sidebar.selectbox('OS Bit', os_bit_encoder.classes_)
graphic_card_gb = st.sidebar.selectbox('Graphics Card GB', graphic_card_gb_encoder.classes_)
weight = st.sidebar.selectbox('Weight', weight_encoder.classes_)
warranty = st.sidebar.selectbox('Warranty', warranty_encoder.classes_)
Touchscreen = st.sidebar.selectbox('Touch Screen', touchscreen_encoder.classes_)
msoffice = st.sidebar.selectbox('MS Office', msoffice_encoder.classes_)

brand_encode = brand_encoder.transform([brand])[0]
processor_brand_encode = processor_brand_encoder.transform([processor_brand])[0]
processor_name_encode = processor_name_encoder.transform([processor_name])[0]
processor_gnrtn_encode = processor_gnrtn_encoder.transform([processor_gnrtn])[0]
ram_gb_encode = ram_gb_encoder.transform([ram_gb])[0]
ram_type_encode = ram_type_encoder.transform([ram_type])[0]
ssd_encode = ssd_encoder.transform([ssd])[0]
hdd_encode = hdd_encoder.transform([hdd])[0]
os_encode = os_encoder.transform([os])[0]
os_bit_encode = os_bit_encoder.transform([os_bit])[0]
graphic_card_gb_encode = graphic_card_gb_encoder.transform([graphic_card_gb])[0]
weight_encode = weight_encoder.transform([weight])[0]
warranty_encode = warranty_encoder.transform([warranty])[0]
touchscreen_encode = touchscreen_encoder.transform([Touchscreen])[0]
msoffice_encode = msoffice_encoder.transform([msoffice])[0]

total = {
    'brand': [brand_encode],
    'processor_brand': [processor_brand_encode],
    'processor_name': [processor_name_encode],
    'processor_gnrtn': [processor_gnrtn_encode],
    'ram_gb': [ram_gb_encode],
    'ram_type': [ram_type_encode],
    'ssd': [ssd_encode],
    'hdd': [hdd_encode],
    'os': [os_encode],
    'os_bit': [os_bit_encode],
    'graphic_card_gb': [graphic_card_gb_encode],
    'weight': [weight_encode],
    'warranty': [warranty_encode],
    'Touchscreen': [touchscreen_encode],
    'msoffice': [msoffice_encode]
}

features = pd.DataFrame(total)
dt = pd.DataFrame(total)
st.dataframe(dt, width=900)

if st.button('Check'):
    prediction = model.predict(features)
    # st.write('Price Range', prediction[0])
    st.write(f'Price range prediction is $ {prediction[0]}')
