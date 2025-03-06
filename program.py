import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold,GridSearchCV, cross_validate,cross_val_score
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
from joblib import dump,load

# Membuat Sidebar
if "main_select" not in st.session_state:
    st.session_state["main_select"] = "Penyakit Tiroid"  
if "smote_option" not in st.session_state:
    st.session_state["smote_option"] = None
def reset_main_select():
    st.session_state["main_select"] = None
def reset_smote_option():
    st.session_state["smote_option"] = None

menu_main = st.sidebar.selectbox(
    "Pilih Menu Utama",
    ["Penyakit Tiroid", "Dataset", "Preprocessing"],
    key="main_select",
    on_change=reset_smote_option
)

menu_smote = st.sidebar.selectbox(
    "Pilih data Training",
    ["Tanpa SMOTE", "Dengan SMOTE"],
    key="smote_option",
    on_change=reset_main_select
)
if st.session_state["main_select"]and not st.session_state["smote_option"]:
    if st.session_state["main_select"] == "Penyakit Tiroid":
        # st.markdown("<h3 style='text-align: center;'>Penyakit Tiroid</h3>", unsafe_allow_html=True)
        gambar = "https://img.freepik.com/premium-vector/thyroid-human-organ-isolated-white-background-vector-illustration-flat-design_545793-366.jpg"
        st.markdown(
        f"""
            <div style="text-align: center;">
                <img src="{gambar}" style="width: 50%; height: auto;">
            </div>
        """,
        unsafe_allow_html=True
    )
        st.markdown("""Tiroid atau yang dikenal dengan kelenjar gondok merupakan kelenjar yang terdapat pada bagian leher tepatnya di bawah jakun dengan bentuk yang dimiliki hampir sama seperti kupu-kupu. Kelenjar tiroid ini sangat berpengaruh dalam tubuh manusia karena fungsi kelenjar tiroid ini adalah menghasilkan hormon tiroid untuk merangsang pembentukan protein dalam tubuh dengan meningkatkan jumlah oksigen oleh sel tubuh. Namun, meskipun begitu adanya gangguan hormon tiroid yang bekerja dapat menyebabkan seseorang terkena penyakit hipertiroid yaitu kelebihan hormon tiroid sedangkan hipotiroid kurangnya hormon tiroid dalam tubuh.""", unsafe_allow_html=True)
        st.markdown("""Jumlah data yang mengalami penyakit tiroid di dunia berdasarakn kementrian kesehatan dunia masih tergolong tinggi pada bayi baru lahir yang terkena hiperthyroid dengan angka 1600 tiap tahun dan banyak didominasi oleh perempuan dengan prevalensi terjadi 4 hingga 10 kali dibandingkan pria dengan angka persentase 14,7% berdasarkan data dari Kemenkes RI tahun 2015.""", unsafe_allow_html=True)

    elif st.session_state["main_select"]=="Dataset":
        st.title("Dataset Penyakit Tiroid")
        link = "tiroid..csv"
        dataset = pd.read_csv(link, header="infer", index_col=False,  sep=";")
        st.dataframe(dataset)
        st.markdown(
            f"""Data yang digunakan pada klasifikasi penyakit tiroid ini diambil dari website resmi UC Irvine Machine Learning pada https://archive.ics.uci.edu/dataset/102/thyroid+disease yang diambil pada bulan Februari 2024 dengan jumlah data sebanyak {len(dataset)} data, {len(dataset.columns)} fitur. Terdiri dari 3 kelas yaitu normal, hyperthyroid, dan hypotiroid.
            """, unsafe_allow_html=True)
            # st.write(dataset["referral_source"].value_counts())
        kolom=(list(dataset.columns))
        
        deskripsi ={
            'age' : 'Umur Pasien',
            'sex' : 'Jenis Kelamin Pasien',
            'query_on_thyroxine' : 'Permintaan penggunaan obat levothyroxin merupakan obat yang menggantikan kurangnya produksi hormon tiroid dalam tubuh (thyroxine) ',
            'on_thyroxine' : 'Pasien sedang menggunakan obat levothyroxine merupakan obat yang menggantikan kurangnya produksi hormon tiroid dalam tubuh (thyroxine)',
            'on_antithyroid_meds' : 'Pasien sedang menggunakan obat antitiroid yaitu obat untuk mengobati penyakit tiroid',
            'sick' : 'Pasien sedang mengalami sakit',
            'pregnant' : 'Pasien sedang dalam masa kehamilan',
            'thyroid_surgery' : 'Pasien sedang menjalani operasi tiroid',
            'I131_treatment' : 'Pengobatan penyakit tiroid dengan memanfaatkan aktifitas kelenjar tiroid dengan yodium dengan tujuan untuk menghancurkan jaringan tiroid',
            'query_hypothyroid' : 'Kurangnya produksi hormon tiroid oleh kelenjar tiroid',
            'query_hyperthyroid' : 'Produksi hormon tiroid oleh kelenjar tiroid yang berlebihan',
            'lithium' : 'Obat yang digunakan untuk penyembuhan penyakit tiroid dan sebagai terapi tambahan dalam penanganan hipertirodisme berat',
            'goitre' : 'Pasien sedang menderita penyakit gondok yang disebabkan karena pembengkakan kelenjar tiroid',
            'tumor' : 'kelenjar tiroid mengalami kelainan berupa pembesaran',
            'hypopituitary' : 'Kurangnya kelenjar tiroid dalam memproduksi hormon tiroksin yang cukup ',
            'psych' : 'Pasien mengalami adanya gangguan pada kesehatan mental',
            'TSH' : 'Tingkat TSH (Thyroid Stimulating Hormone) hormon perangsang tiroid pada Pasien',
            'TSH_measured' : 'Dilakukan pemeriksaan lebih lanjut hormon TSH dalam darah',
            'T3' : 'Tingkat hormon  triiodotironin (tiroksin) yang diproduksi pada kelenjar tiroid dan berperan penting dalam metabolisme dalam tubuh',
            'T3_measured' : 'Dilakukan pemeriksaan hormon T3 dalam darah',
            'FT4' : 'Bagian dari hormon T4(Thyroxine)  yang tidak terikat dengan protein pengikat dalam darah. Bertujuan untuk mengukur jumlah hormon T4(Thyroxine) bebas',
            'FT4_measured' : 'Dilakukan pemeriksaan hormon T4(Thyroxine) yang tidak terikat dengan protein pengikat dalam darah',
            'T4' : 'Tingkat hormon T4 (Thyroxine) yang diproduksi oleh kelenjar tiroid  dan berperan penting dalam metabolisme dalam tubuh',
            'T4_measured' : 'Dilakukan pemeriksaan hormon T4(Thyroxine) dalam darah',
            'FTI' : 'Tingkat FTI (Free Thyroxine Index) atau Indeks Tiroid Bebas merupakan indikator status tiroid',
            'FTI_measured' : 'Dilakukan pemeriksaan FTI dalam darah pada pasien',
            'TBG' : 'Tingkat TBG (Thyroxine-binding globulin) yaitu protein yang berikatan dengan T3 maupun T4 di dalam darah dengan afinitas yang tinggi. TBG bertujuan untuk mengikat dan mengangkut hormon tiroid ke seluruh tubuh',
            'TBG_measured' : 'Dilakukan pemeriksaan TBG dalam darah',
            'referral_source' : 'Sumber rujukan pasien untuk melakukan konsultasi',
            'patient_id' : 'Kode unik untuk pasien',
            'target': 'Mendiagnosis penyakit tiroid pada pasien'
        }
        if "muncul" not in st.session_state:
            st.session_state["muncul"]=False
        def tombol():
            st.session_state["muncul"]= not st.session_state["muncul"]
        if st.button("Keterangan"):
            tombol()
        if st.session_state["muncul"]:
            for col in kolom:
                desc = deskripsi.get(col, '-')
                st.write(f"- **{col}**: {desc}")

        
        st.write("Fitur yang memiliki data kosong : ",dataset.isnull().sum())
        # else:  
        #     st.write("Upload file terlebih dahulu")  


    elif st.session_state["main_select"] == "Preprocessing":
        st.title("Preprocessing dataset")
        link = "tiroid..csv"
        dataset = pd.read_csv(link, header="infer", index_col=False,  sep=";")
        st.write("Data Sebelum Preprocessing")
        st.write("Jumlah Fitur", len(dataset.columns))
        st.write(dataset)
        dataset['target'] = dataset['target'].replace({'-': 0,'A': 1,'B': 1,'C': 1,'D': 1,'E': 2,'F': 2,'G': 2,'H': 2})

        kolom_one_hot = ["referral_source"]
        kolom_label = ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick", 
                "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", 
                "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", 
                "psych", "TSH_measured", "T3_measured", "FT4_measured", 
                "T4_measured", "FTI_measured", "TBG_measured"]


        encode = OneHotEncoder(sparse_output=False, drop=None)  
        dataset_one_hot = pd.DataFrame(encode.fit_transform(dataset[kolom_one_hot]))
        kolom_baru = encode.get_feature_names_out(kolom_one_hot)
        dataset_one_hot.columns = kolom_baru
        dataset = dataset.drop(columns=kolom_one_hot)
        dataset = pd.concat([dataset, dataset_one_hot], axis=1)
        dump(encode, 'onehot_encoder.pkl')


        label_encod = LabelEncoder()
        for column in kolom_label:
            dataset[column] = label_encod.fit_transform(dataset[column])
        dataset["target"] = label_encod.fit_transform(dataset["target"])
        dataset["sex"].replace(2,0)
        dump(label_encod, 'label_encoder.pkl')
        st.write("Hasil Transformasi Data :")
        st.write(dataset)

        #imputasi
        kolom = ["sex", "TSH", "T3", "FT4", "T4", "FTI", "TBG"]
        imputer = KNNImputer(n_neighbors=3)
        dataset[kolom] = imputer.fit_transform(dataset[kolom])
        dump(imputer, 'imputer.pkl')
        st.write("Hasil Imputasi Data dengan Imputasi KNN :")
        st.write(dataset)

        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        kolom_numerik = ["age", "TSH", "T3", "FT4", "T4", "FTI", "TBG","patient_id"]
        kolom_minmax = ["age","sex","TSH","T3","FT4","T4","FTI","TBG","patient_id"]
        dataset_ori = X[kolom_minmax]
        dump(dataset_ori,"dataset_ori.pkl")
        #st.write("Dump berhasil")
        scaler = MinMaxScaler()
        X[kolom_numerik] = scaler.fit_transform(X[kolom_numerik])
        dump(scaler, 'scaler.pkl')
        dataset = pd.concat([X, y], axis=1)
        st.write("Hasil Normalisasi :")
        st.write("Jumlah Fitur", len(dataset.columns))
        st.write(dataset)

if st.session_state["smote_option"] and not st.session_state["main_select"]:
    if st.session_state["smote_option"] == "Tanpa SMOTE":
        link = "tiroid..csv"
        dataset = pd.read_csv(link, header="infer", index_col=False,  sep=";")
        tanpa_smote,ig,uji,hasil_terbaik= st.tabs(["Data Tanpa SMOTE", "Seleksi Information Gain", "Metode LS-SVM","Hasil Terbaik"])
        
        dataset['target'] = dataset['target'].replace({'-': 0,'A': 1,'B': 1,'C': 1,'D': 1,'E': 2,'F': 2,'G': 2,'H': 2})
        kolom_one_hot = ["referral_source"]
        kolom_label = ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick", 
                "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", 
                "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", 
                "psych", "TSH_measured", "T3_measured", "FT4_measured", 
                "T4_measured", "FTI_measured", "TBG_measured"]


        encode = OneHotEncoder(sparse_output=False, drop=None)  
        dataset_one_hot = pd.DataFrame(encode.fit_transform(dataset[kolom_one_hot]))
        kolom_baru = encode.get_feature_names_out(kolom_one_hot)
        dataset_one_hot.columns = kolom_baru
        dataset = dataset.drop(columns=kolom_one_hot)
        dataset = pd.concat([dataset, dataset_one_hot], axis=1)
        dataset["sex"] = dataset["sex"].replace(2, 0)
        label_encod = LabelEncoder()
        for column in kolom_label:
            dataset[column] = label_encod.fit_transform(dataset[column])
        #imputasi
        kolom = ["sex", "TSH", "T3", "FT4", "T4", "FTI", "TBG"]
        imputer = KNNImputer(n_neighbors=5)
        dataset[kolom] = imputer.fit_transform(dataset[kolom])

        # kolom_numerik = ["age", "TSH", "T3", "FT4", "T4", "FTI", "TBG"] 
        # scaler = MinMaxScaler()
        # dataset[kolom_numerik] = scaler.fit_transform(dataset[kolom_numerik])
        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        kolom_numerik = ["age", "TSH", "T3", "FT4", "T4", "FTI", "TBG","patient_id"]
        scaler = MinMaxScaler()
        X[kolom_numerik] = scaler.fit_transform(X[kolom_numerik])
        dataset = pd.concat([X, y], axis=1)

        with tanpa_smote:
            #slider tanpa smote
            test_ts = st.slider("Data testing tanpa smote :",0.0,1.0,0.9 ,step=0.1)
            X = dataset.drop(columns=["target"])
            y = dataset["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ts, random_state=42, stratify=y)
            st.write(X_train)

            jumlah_kelas = Counter(y_train)
            fig, axes = plt.subplots(figsize=(4, 2))
            axes.bar(Counter(y_train).keys(), Counter(y_train).values(), color='lightblue',width=0.3)
            axes.set_title('Grafik Data Tanpa SMOTE')
            axes.set_xlabel('Kelas')
            axes.set_ylabel('Jumlah')
            axes.set_xticks(list(jumlah_kelas.keys()))
            st.pyplot(fig)

            a,b = st.columns(2)
            with a :
                st.write("Data Latih : ",y_train.count())
                st.write("Data uji : ",y_test.count())
            with b :
                st.write("Jumlah setiap kelas : ",Counter(y_train))

        with ig:
            #seleksi ig
            quantile = st.slider("Quantile ", 0.0, 1.0, 0.75, step=0.05) 
            mutual_info = mutual_info_classif(X_train, y_train)
            seleksi_ig = pd.DataFrame({"Fitur": X_train.columns, "Information Gain": mutual_info})
            st.write("Nilai gain pada setiap fitur :")
            seleksi_ig
            threshold = seleksi_ig["Information Gain"].quantile(quantile)
            st.write("Nilai Threshold ",threshold)
            seleksi_fitur = seleksi_ig[seleksi_ig["Information Gain"] >= threshold]
            st.write(f"Fitur yang dipilih berdasarkan (Quantile  >= {quantile})  :")
            seleksi_fitur
            seleksi_kolom = list(dict.fromkeys(seleksi_fitur["Fitur"].to_list()))
            # column_order = X_train[selected_features].columns.tolist()  
            with open('selected_features.pkl', 'wb') as f:
                pickle.dump(seleksi_kolom, f)
            # X_train_ig = X_train[seleksi_kolom]
            # X_test_ig = X_test[seleksi_kolom]

            # scaler_ig = MinMaxScaler()
            # X_train_ig = scaler_ig.fit_transform(X_train_ig)
            # X_test_ig = scaler_ig.transform(X_test_ig)
            # dump(scaler_ig, 'scaler_ig.pkl')


            st.session_state.seleksi_fitur = seleksi_fitur

            st.write("Sebelum Dilakukan Seleksi Fitur :")            
            st.dataframe(dataset)
            st.write("Jumlah fitur : ", len(dataset.columns))
            
            #hasil fitur seleksi
            st.write("Setelah Dilakukan Seleksi Fitur :")
            seleksi_fitur = seleksi_fitur["Fitur"]
            data_ig = X_train.loc[:, seleksi_fitur]
            data_ig = pd.concat([data_ig.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

            scaler_ig = MinMaxScaler()
            X_train_ig = scaler_ig.fit_transform(data_ig.drop(columns="target"))
            X_test_ig = scaler_ig.transform(X_test[seleksi_fitur])
            dump(scaler_ig, 'scaler_ig.pkl')

            # data_ig = pd.DataFrame(X_train_ig, columns=seleksi_kolom)
            # data_ig["target"] = y_train.values
            # st.write("Data Setelah Seleksi Fitur:")
            # st.dataframe(data_ig)
            # st.write("Jumlah fitur : ", len(data_ig.columns))
            

            presentase =  ((len(dataset.columns)-len(data_ig.columns)) / len(dataset.columns))*100
            st.write(f"Presentase pengurangan fitur :{presentase:.2f}%")

        
        if 'best_results' not in st.session_state:  
            st.session_state.best_results = None 
        with uji:
            class LSSVM(BaseEstimator, ClassifierMixin):
                def __init__(self, C=None, kernel=None, gamma=None):
                    self.C = C
                    self.kernel = kernel
                    self.gamma = gamma
                    self.models = {}

                @staticmethod
                def kernel_linear(xi, xj):
                    return np.dot(xi.astype(np.float32), xj.T.astype(np.float32))

                @staticmethod
                def kernel_rbf(xi, xj, gamma):
                    xi = np.atleast_2d(xi)
                    xj = np.atleast_2d(xj)
                    return np.exp(-gamma * cdist(xi, xj, 'sqeuclidean'))

                def fit(self, X, y):
                    self.classes_ = np.unique(y)
                    self.kernel_func = self.fungsi_kernel()  

                    for cls in self.classes_:
                        y_biner = np.where(y == cls, 1, -1).astype(np.float32)
                        model = self.train_binary_lssvm(X, y_biner)
                        self.models[cls] = model
                    return self

                def train_binary_lssvm(self, X, y_biner):
                    if self.kernel == "rbf":
                        omega = self.kernel_func(X, X, self.gamma)
                    else:
                        omega = self.kernel_func(X, X).astype(np.float32)
                    
                    ones = np.ones((len(y_biner), 1), dtype=np.float32)
                    omega += np.eye(len(y_biner), dtype=np.float32) * 1e-12  
                    alpha = np.block([[0, ones.T], [ones, omega + np.eye(len(y_biner), dtype=np.float32) / self.C]])
                    alpha += np.eye(alpha.shape[0]) * 1e-12
                    b = np.concatenate(([0], y_biner))
                    
                    try :
                        solusi = np.linalg.solve(alpha, b)
                    except np.linalg.LinAlgError:
                        solusi = np.linalg.pinv(alpha) @ b
                        
                    intercept = solusi[0]
                    koefisien = solusi[1:]
                    return {"intercept": intercept, "koefisien": koefisien, "support_vector": X}

                def predict(self, X):
                    hasil_prediksi = []
                    for cls, model in self.models.items():
                        intercept = model["intercept"]
                        koefisien = model["koefisien"]
                        support_vector = model["support_vector"]
                        if self.kernel == "rbf":
                            hasil_kernel = self.kernel_func(X, support_vector, self.gamma)
                        else:
                            hasil_kernel = self.kernel_func(X, support_vector)
                        prediksi = hasil_kernel @ koefisien + intercept
                        hasil_prediksi.append(prediksi)
                    
                    hasil_prediksi = np.array(hasil_prediksi).T
                    return self.classes_[np.argmax(hasil_prediksi, axis=1)]

                def fungsi_kernel(self):
                    if self.kernel == "linear":
                        return self.kernel_linear
                    elif self.kernel == "rbf":
                        return self.kernel_rbf
           
            sken = st.selectbox("Skenario", ["Skenario Tanpa Tuning","Skenario dengan Tuning"])

            if sken =="Skenario Tanpa Tuning":
                st.markdown(f"""<div style="text-align: center;"><h5>Test Uji Tanpa SMOTE dan Tanpa Tuning Grid Search</div>""",unsafe_allow_html=True)

                param_c = st.number_input("Parameter Regulasi C")
                param_gamma = st.number_input("Parameter Kernel Gamma")
                k_fold = 5  
                
                skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)  
                
                opsi_data = st.selectbox("Pilih Skenario Data:",["Tanpa Seleksi Fitur", "Dengan Seleksi Fitur"])
                if opsi_data == "Tanpa Seleksi Fitur":
                    X_train = X_train
                    y_train = y_train
                    X_test = X_test
                    y_test = y_test
                else:
                    with open('selected_features.pkl', 'rb') as f:
                        selected_features = pickle.load(f)
                    # X_train = data_ig[seleksi_fitur]
                    # y_train = data_ig["target"]
                    # X_test = X_test[seleksi_fitur]
                    # y_test = y_test
                    X_train = X_train[selected_features]  
                    y_train = y_train
                    X_test = X_test[selected_features]
                    y_test = y_test
                
                if st.button("Test Uji"):
                    klasif_linear = LSSVM(C=param_c, kernel="linear")    
                    waktu_awal = time.time()    
                    klasif_linear.fit(X_train, y_train)    
                    
                    # Cross-validation    
                    hasil_fold_linear = []  
                    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):  
                        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]  
                        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]  
                        
                        klasif_linear.fit(X_train_fold, y_train_fold)  
                        y_pred_fold = klasif_linear.predict(X_test_fold)  
                        
                        akurasi_fold = accuracy_score(y_test_fold, y_pred_fold)  
                        presisi_fold = precision_score(y_test_fold, y_pred_fold, average='weighted')  
                        recall_fold = recall_score(y_test_fold, y_pred_fold, average='weighted')  
                        f1_fold = f1_score(y_test_fold, y_pred_fold, average='weighted')  
                        
                        hasil_fold_linear.append({  
                            "Fold": fold + 1,  
                            "Akurasi": akurasi_fold,  
                            "Presisi": presisi_fold,  
                            "Recall": recall_fold,  
                            "F-1 Score": f1_fold  
                        })  
                    
                    waktu_akhir = time.time()    
                    waktu_eksekusi_linear = waktu_akhir - waktu_awal    
                    y_prediksi_linear = klasif_linear.predict(X_test)    
                    
                    # Menghitung metrik rata-rata  
                    rata_akurasi_linear = np.mean([fold['Akurasi'] for fold in hasil_fold_linear])  
                    presisi_linear = np.mean([fold['Presisi'] for fold in hasil_fold_linear])  
                    recall_linear = np.mean([fold['Recall'] for fold in hasil_fold_linear])  
                    f1score_linear = np.mean([fold['F-1 Score'] for fold in hasil_fold_linear])  
                    
                    # RBF Kernel    
                    klasif_rbf = LSSVM(C=param_c, kernel="rbf", gamma=param_gamma)    
                    waktu_awal = time.time()    
                    klasif_rbf.fit(X_train, y_train)    
                    
                    # Cross-validation    
                    hasil_fold_rbf = []  
                    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):  
                        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]  
                        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]  
                        
                        klasif_rbf.fit(X_train_fold, y_train_fold)  
                        y_pred_fold = klasif_rbf.predict(X_test_fold)  
                        
                        akurasi_fold = accuracy_score(y_test_fold, y_pred_fold)  
                        presisi_fold = precision_score(y_test_fold, y_pred_fold, average='weighted')  
                        recall_fold = recall_score(y_test_fold, y_pred_fold, average='weighted')  
                        f1_fold = f1_score(y_test_fold, y_pred_fold, average='weighted')  
                        
                        hasil_fold_rbf.append({  
                            "Fold": fold + 1,  
                            "Akurasi": akurasi_fold,  
                            "Presisi": presisi_fold,  
                            "Recall": recall_fold,  
                            "F-1 Score": f1_fold  
                        })  
                       
                    waktu_akhir = time.time()    
                    waktu_eksekusi_rbf = waktu_akhir - waktu_awal    
                    y_prediksi_rbf = klasif_rbf.predict(X_test)    
                    
                    # rata-rata
                    rata_akurasi_rbf = np.mean([fold['Akurasi'] for fold in hasil_fold_rbf])  
                    presisi_rbf = np.mean([fold['Presisi'] for fold in hasil_fold_rbf])  
                    recall_rbf = np.mean([fold['Recall'] for fold in hasil_fold_rbf])  
                    f1score_rbf = np.mean([fold['F-1 Score'] for fold in hasil_fold_rbf])  
                    
                    #fold terbaik
                    terbaik_linear = max(hasil_fold_linear, key=lambda x: x['Akurasi'])  
                    terbaik_rbf = max(hasil_fold_rbf, key=lambda x: x['Akurasi']) 
                    
                    # Hasil Evaluasi    
                    hasil_linear = {    
                        "Kernel": "Linear",    
                        "Akurasi": rata_akurasi_linear,    
                        "Presisi": presisi_linear,    
                        "Recall": recall_linear,    
                        "F-1 Score": f1score_linear,
                        "waktu ":  waktu_eksekusi_linear    
                    }    
                    hasil_rbf = {    
                        "Kernel": "RBF",    
                        "Akurasi": rata_akurasi_rbf,    
                        "Presisi": presisi_rbf,    
                        "Recall": recall_rbf,    
                        "F-1 Score": f1score_rbf,
                        "waktu ":  waktu_eksekusi_rbf    
                    }    
                    st.session_state.best_results = {"Linear": hasil_linear, "RBF": hasil_rbf}     
                    
                    if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:    
                        st.write("Fitur Terseleksi:")    
                        st.write(st.session_state.seleksi_fitur)  
                    
                    # Confusion Matrix  
                    cm_linear = confusion_matrix(y_test, y_prediksi_linear)  
                    cm_rbf = confusion_matrix(y_test, y_prediksi_rbf)  
                    
                    kol1, kol2 = st.columns(2)  
                    with kol1:  
                        fig, ax = plt.subplots()  
                        ConfusionMatrixDisplay(confusion_matrix=cm_linear, display_labels=np.unique(y_test)).plot(ax=ax, cmap='Blues', colorbar=True)  
                        st.pyplot(fig)  
                    with kol2:  
                        fig, ax = plt.subplots()  
                        ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=np.unique(y_test)).plot(ax=ax, cmap='Blues', colorbar=True)  
                        st.pyplot(fig)   
                    if hasil_rbf["Akurasi"] > hasil_linear["Akurasi"]:
                        st.session_state.best_results = hasil_rbf
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_prediksi_rbf)
                    else:
                        st.session_state.best_results = hasil_linear
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_prediksi_linear)

                    st.write("Hasil Terbaik:")  
                    st.table(pd.DataFrame([terbaik_linear, terbaik_rbf]))
                    
                    st.write("Hasil Rata-Rata:")  
                    st.table(pd.DataFrame([hasil_linear, hasil_rbf]))     
                    
                    # Visualisasi Metrik  
                    metrik = ["Akurasi", "Presisi", "Recall", "F-1 Score"]  
                    met_linear = [hasil_linear['Akurasi'], hasil_linear['Presisi'], hasil_linear['Recall'], hasil_linear['F-1 Score']]   
                    met_rbf = [hasil_rbf['Akurasi'], hasil_rbf['Presisi'], hasil_rbf['Recall'], hasil_rbf['F-1 Score'] ]  
                    
                    fig, ax = plt.subplots(figsize=(6, 4))  
                    x = np.arange(len(metrik))    
                    width = 0.3    
                    bars1 = ax.bar(x - width/2, met_linear, width, label='Linear', color='lightblue')  
                    bars2 = ax.bar(x + width/2, met_rbf, width, label='RBF', color='#A7D477')  
            
                    linear_patch = mpatches.Patch(color='lightblue', label='Kernel Linear')  
                    rbf_patch = mpatches.Patch(color='#A7D477', label='Kernel RBF')  
                    ax.legend(handles=[linear_patch, rbf_patch], loc='upper right')  
            
                    ax.set_xlabel('Metrik Evaluasi')  
                    ax.set_ylabel('Nilai Evaluasi')  
                    ax.set_title('Perbandingan Kinerja Model')  
                    ax.set_xticks(x)  
                    ax.set_xticklabels(metrik)  
                    st.pyplot(fig)  
                    
                    if sken == "Skenario Tanpa Tuning":  
                        if opsi_data == "Tanpa Seleksi Fitur":  
                            dump(klasif_linear, "linear_tanpa_ig_tanpaSmote.pkl")  
                            dump(klasif_rbf, "rbf_tanpa_ig_tanpaSmote.pkl")  
                        else:
                            dump(klasif_linear, "linear_ig1_tanpaSmote.pkl")  
                            dump(klasif_rbf, "rbf_ig1_tanpaSmote.pkl")  
                
            else:
                st.markdown("""<div style="text-align: center;"><h5>Test Uji Tanpa SMOTE dengan Tuning Grid Search</h5></div>""",unsafe_allow_html=True,)

                param_c = st.text_input("Masukkan nilai parameter C (pisahkan dengan koma, contoh: 0.1,10,100)", "0.1,10,100")
                param_c = [float(c.strip()) for c in param_c.split(",")]

                param_gamma = st.text_input("Masukkan nilai parameter gamma (pisahkan dengan koma, contoh: 1,100,1000)", "1,100,1000")
                param_gamma = [float(g.strip()) for g in param_gamma.split(",")]

                k_fold = 5
                skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)

                # Pilihan data
                opsi_data = st.selectbox("Pilih Skenario Data:", ["Tanpa Seleksi Fitur", "Dengan Seleksi Fitur"])
                if opsi_data == "Tanpa Seleksi Fitur":
                    X_train = X_train
                    y_train = y_train
                    X_test = X_test
                    y_test = y_test
                else:
                    X_train = data_ig[seleksi_fitur]
                    y_train = data_ig["target"]
                    X_test = X_test[seleksi_fitur]
                    y_test = y_test

                # Test Uji
                if st.button("Test Uji"):
                    param_grid_linear = {"C": param_c}
                    param_grid_rbf = {"C": param_c, "gamma": param_gamma}

                    # Kernel linear  
                    klasif_linear = GridSearchCV(  
                        estimator=LSSVM(kernel="linear"),  
                        param_grid=param_grid_linear,  
                        cv=skf,  
                        scoring="accuracy",  
                    )  
                    start_time = time.time()  
                    klasif_linear.fit(X_train, y_train)  
                    waktu_linear = time.time() - start_time  
                
                    best_params_linear = klasif_linear.best_params_  
                    best_score_linear = klasif_linear.best_score_  
                    y_pred_linear = klasif_linear.best_estimator_.predict(X_test)  
                
                    # RBF Kernel  
                    klasif_rbf = GridSearchCV(  
                        estimator=LSSVM(kernel="rbf"),  
                        param_grid=param_grid_rbf,  
                        cv=skf,  
                        scoring="accuracy",  
                    )  
                    start_time = time.time()  
                    klasif_rbf.fit(X_train, y_train)  
                    waktu_rbf = time.time() - start_time  
                
                    best_params_rbf = klasif_rbf.best_params_  
                    best_score_rbf = klasif_rbf.best_score_  
                    y_pred_rbf = klasif_rbf.best_estimator_.predict(X_test)  
                

                    hasil_linear = {  
                        "Kernel": "Linear",  
                        "C Terbaik": best_params_linear["C"],  
                        "Akurasi": best_score_linear,  
                        "Presisi": precision_score(y_test, y_pred_linear, average='weighted'),  
                        "Recall": recall_score(y_test, y_pred_linear, average='weighted'),  
                        "F1-Score": f1_score(y_test, y_pred_linear, average='weighted'),  
                        "Waktu ":waktu_linear
                    }  

                    hasil_rbf = {  
                        "Kernel": "RBF",  
                        "C Terbaik": best_params_rbf["C"],  
                        "Gamma Terbaik": best_params_rbf["gamma"],  
                        "Akurasi": best_score_rbf,  
                        "Presisi": precision_score(y_test, y_pred_rbf, average='weighted'),  
                        "Recall": recall_score(y_test, y_pred_rbf, average='weighted'),  
                        "F1-Score": f1_score(y_test, y_pred_rbf, average='weighted'),
                        "Waktu ":waktu_rbf  
                    }  
                    st.session_state.best_results = {"Linear":hasil_linear,"RBF": hasil_rbf}


                    if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:  
                        st.write("Fitur Terseleksi:")  
                        st.session_state.seleksi_fitur 
                    # Confusion Matrix
                    st.write("Confusion Matrix:")
                    col1, col2 = st.columns(2)
                    with col1:
                        cm_linear = confusion_matrix(y_test, y_pred_linear)
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(confusion_matrix=cm_linear).plot(ax=ax, cmap="Blues")
                        st.pyplot(fig)
                    with col2:
                        cm_rbf = confusion_matrix(y_test, y_pred_rbf)
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(confusion_matrix=cm_rbf).plot(ax=ax, cmap="Blues")
                        st.pyplot(fig)

                    if best_score_rbf > best_score_linear:
                        st.session_state.best_results = hasil_rbf
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_pred_rbf)
                    else:
                        st.session_state.best_results = hasil_linear
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_pred_linear)

                    st.write("Hasil Evaluasi Model:")
                    metrik = ["Akurasi","Presisi","Recall","F1-Score"]
                    met_linear = [hasil_linear['Akurasi'],hasil_linear['Presisi'],hasil_linear['Recall'],hasil_linear['F1-Score']] 
                    met_rbf = [hasil_rbf['Akurasi'],hasil_rbf['Presisi'],hasil_rbf['Recall'],hasil_rbf['F1-Score'] ]
                    
                    
                    st.table(pd.DataFrame([hasil_linear, hasil_rbf])) 

                    import matplotlib.patches as mpatches
                    fig, ax = plt.subplots(figsize=(6,4))
                    x = np.arange(len(metrik))  
                    width = 0.3  
                    bars1 = ax.bar(x - width/2, met_linear, width, label='Linear', color='lightblue')
                    bars2 = ax.bar(x + width/2, met_rbf, width, label='RBF', color='#A7D477')

                    linear_patch = mpatches.Patch(color='lightblue', label='Kernel Linear')
                    rbf_patch = mpatches.Patch(color='#A7D477', label='Kernel RBF')
                    ax.legend(handles=[linear_patch, rbf_patch], loc='upper right')

                    ax.set_xlabel('Metrik Evaluasi')
                    ax.set_ylabel('Nilai Evaluasi')
                    ax.set_xticks(x)
                    ax.set_xticklabels(metrik)
                    st.pyplot(fig)
                    if sken =="Skenario Dengan Tuning":
                        if opsi_data == "Dengan Seleksi Fitur":
                            dump(klasif_linear.best_estimator_, "linear_dengan_ig_tanpaSmote.pkl")
                            dump(klasif_rbf.best_estimator_, "rbf_dengan_ig_tanpaSmote.pkl")
                    # st.success("Model berhasil disimpan!")
                    # st.write(f"Vector:{klasif_rbf.best_estimator_.models[0]['support_vector'].shape}")

        with hasil_terbaik:
            if st.session_state.best_results is not None:   
                if sken == "Skenario Tanpa Tuning":
                    st.write("Hasil Terbaik dari Pengujian Sebelumnya Tanpa Menggunakan Tuning") 
                    if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:  
                            st.write("Fitur Terseleksi:")  
                            st.session_state.seleksi_fitur 
                    st.write(f"Confusion Matrix untuk {st.session_state.best_results['Kernel']}:")
                    fig, ax = plt.subplots(figsize=(4, 2))
                    ConfusionMatrixDisplay(confusion_matrix=st.session_state.best_confusion_matrix).plot(ax=ax, cmap='Blues', colorbar=True)
                    st.pyplot(fig)

                    # Tampilkan hasil terbaik
                    st.table(pd.DataFrame([st.session_state.best_results]))
                else:
                    st.write("Hasil Terbaik dari Pengujian Sebelumnya Dengan Menggunakan Tuning") 
                    if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:  
                        st.write("Fitur Terseleksi:")  
                        st.session_state.seleksi_fitur 

                    st.write(f"Confusion Matrix untuk {st.session_state.best_results['Kernel']}:")  
                    fig, ax = plt.subplots(figsize=(4, 2))  
                    ConfusionMatrixDisplay(confusion_matrix=st.session_state.best_confusion_matrix, display_labels=np.unique(y_test)).plot(ax=ax, cmap='Blues', colorbar=True)  
                    st.pyplot(fig) 

                    # Tampilkan hasil terbaik
                    st.write("Hasil Evaluasi Terbaik:")  
                    st.table(pd.DataFrame([st.session_state.best_results])) 
            else:  
                st.write("Belum ada hasil pengujian yang disimpan.")  

        klasif_linear = LSSVM(C=param_c, kernel="linear") 
        klasif_rbf = LSSVM(C=param_c, kernel="rbf", gamma=param_gamma)      

        model_info = {
            'selected_features': seleksi_kolom,
            'scaler': scaler,
            'label_encoder': label_encod,
            'one_hot_encoder': encode,
            'imputer': imputer,
            'model': klasif_linear,  # atau model terbaik yang dipilih
            'kolom_numerik': kolom_numerik,
            'kolom_label': kolom_label,
            'kolom_one_hot': kolom_one_hot,
            'threshold_ig': threshold  # tambahkan threshold IG
        }

        # Simpan semua info
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
    


    elif st.session_state["smote_option"] == "Dengan SMOTE":
        #dengan smote
        smote,ig,uji,hasil_terbaik,klasifikasi= st.tabs(["SMOTE", "Seleksi Information Gain", "Metode LS-SVM","Hasil Terbaik","Klasifikasi"])
        link = "tiroid..csv"
        dataset = pd.read_csv(link, header="infer", index_col=False,  sep=";")
        dataset['target'] = dataset['target'].replace({'-': 0,'A': 1,'B': 1,'C': 1,'D': 1,'E': 2,'F': 2,'G': 2,'H': 2})

        kolom_one_hot = ["referral_source"]
        kolom_label = ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick","pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid","query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary","psych", "TSH_measured", "T3_measured", "FT4_measured","T4_measured", "FTI_measured", "TBG_measured"]


        encode = OneHotEncoder(sparse_output=False, drop=None)  
        dataset_one_hot = pd.DataFrame(encode.fit_transform(dataset[kolom_one_hot]))
        kolom_baru = encode.get_feature_names_out(kolom_one_hot)
        dataset_one_hot.columns = kolom_baru
        dataset = dataset.drop(columns=kolom_one_hot)
        dataset = pd.concat([dataset, dataset_one_hot], axis=1)
        label_encod = LabelEncoder()
        for column in kolom_label:
            dataset[column] = label_encod.fit_transform(dataset[column])
        dataset["target"] = label_encod.fit_transform(dataset["target"])
    
        #imputasi
        kolom = ["sex", "TSH", "T3", "FT4", "T4", "FTI", "TBG"]
        imputer = KNNImputer(n_neighbors=3)
        dataset[kolom] = imputer.fit_transform(dataset[kolom])
        
        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        kolom_numerik = ["age", "TSH", "T3", "FT4", "T4", "FTI", "TBG","patient_id"]
        scaler = MinMaxScaler()
        X[kolom_numerik] = scaler.fit_transform(X[kolom_numerik])
        dataset = pd.concat([X, y], axis=1)

        with smote:
            test = st.slider("Jumlah Data Testing :",0.0,1.0,0.9 ,step=0.1)
            X = dataset.drop(columns=["target"])
            y = dataset["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=42, stratify=y)
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            if not isinstance(X_train_smote, pd.DataFrame): #menampilkan hasil smote ke dalam dataframe
                X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
            data_smote = pd.concat([X_train_smote, y_train_smote.reset_index(drop=True)], axis=1)
            data_smote.rename(columns={data_smote.columns[-1]: 'target'}, inplace=True)
            st.dataframe(data_smote)

            fig, axes = plt.subplots(1, 2, figsize=(6, 4))
            axes[0].bar(Counter(y_train).keys(), Counter(y_train).values(), color='lightblue')
            axes[0].set_title('Distribusi Sebelum SMOTE')
            axes[0].set_xlabel('Kelas')
            axes[0].set_ylabel('Jumlah')
            axes[1].bar(Counter(y_train_smote).keys(), Counter(y_train_smote).values(), color='lightgreen')
            axes[1].set_title('Distribusi Sesudah SMOTE')
            axes[1].set_xlabel('Kelas')
            axes[1].set_ylabel('Jumlah')
            plt.tight_layout()
            st.pyplot(plt)
            before, after = st.columns(2)
            with before:
                st.write("Sebelum SMOTE:", Counter(y_train))
                st.write("Jumlah data : ",y_train.count())
            with after:
                st.write("Sesudah SMOTE:", Counter(y_train_smote))
                st.write("Jumlah data : ",y_train_smote.count())

        with ig:
            quantile = st.slider("Pilih Quantile (0.0 - 1.0)", 0.0, 1.0, 0.75, step=0.05) #seleksi fitur ig berdasarakan quantile
            mutual_info = mutual_info_classif(X_train_smote, y_train_smote)
            seleksi_ig = pd.DataFrame({"Fitur": X_train.columns, "Information Gain": mutual_info})
            st.write("Nilai gain pada setiap fitur :")
            seleksi_ig
            threshold = seleksi_ig["Information Gain"].quantile(quantile)
            st.write("Nilai Threshold ",threshold)
            seleksi_fitur = seleksi_ig[seleksi_ig["Information Gain"] >= threshold]
            st.write(f"Fitur yang dipilih berdasarkan (Quantile  >= {quantile})  :")
            seleksi_fitur
            st.session_state.seleksi_fitur=seleksi_fitur
            seleksi_kol = list(dict.fromkeys(seleksi_fitur["Fitur"].to_list())) #menyimpan hasil seleksi fitur ke dalam list

            st.write("Sebelum Dilakukan Seleksi Fitur :")
            X = dataset.drop(columns=["target"])
            y = dataset["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=42, stratify=y)
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            #manmapilkan hasil list smote ke dalam data frame
            if not isinstance(X_train_smote, pd.DataFrame):
                X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
            data_smote = pd.concat([X_train_smote, y_train_smote.reset_index(drop=True)], axis=1)
            data_smote.rename(columns={data_smote.columns[-1]: 'target'}, inplace=True)
            
            st.dataframe(data_smote.head())
            st.write("Jumlah fitur : ", len(data_smote.columns))
            
            #hasil fitur seleksi
            st.write("Setelah Dilakukan Seleksi Fitur :")
            seleksi_fitur = seleksi_fitur["Fitur"]
            data_ig = X_train_smote.loc[:, seleksi_fitur]
            data_ig = pd.concat([data_ig, y_train_smote.reset_index(drop=True)], axis=1)
            st.dataframe(data_ig.head())
            st.write("Jumlah fitur : ", len(data_ig.columns))

            presentase =  ((len(data_smote.columns)-len(data_ig.columns)) / len(data_smote.columns))*100
            st.write(f"Presentase pengurangan fitur :{presentase:.2f}%")

        if 'best_results' not in st.session_state:  
            st.session_state.best_results = None 

        with uji:
            class LSSVM(BaseEstimator, ClassifierMixin):
                def __init__(self, C=None, kernel=None, gamma=None):
                    self.C = C
                    self.kernel = kernel
                    self.gamma = gamma
                    self.models = {}

                @staticmethod
                def kernel_linear(xi, xj):
                    return np.dot(xi.astype(np.float32), xj.T.astype(np.float32))

                @staticmethod
                def kernel_rbf(xi, xj, gamma):
                    xi = np.atleast_2d(xi)
                    xj = np.atleast_2d(xj)
                    return np.exp(-gamma * cdist(xi, xj, 'sqeuclidean'))

                def fit(self, X, y):
                    self.classes_ = np.unique(y)
                    self.kernel_func = self.fungsi_kernel()  # Tentukan fungsi kernel yang digunakan

                    for cls in self.classes_:
                        y_biner = np.where(y == cls, 1, -1).astype(np.float32)
                        model = self.train_binary_lssvm(X, y_biner)
                        self.models[cls] = model
                    
                    return self

                def train_binary_lssvm(self, X, y_biner):
                    if self.kernel == "rbf":
                        omega = self.kernel_func(X, X, self.gamma)
                    else:
                        omega = self.kernel_func(X, X).astype(np.float32)
                    
                    ones = np.ones((len(y_biner), 1), dtype=np.float32)
                    omega += np.eye(len(y_biner), dtype=np.float32) * 1e-12  # Regularization term
                    alpha = np.block([[0, ones.T], [ones, omega + np.eye(len(y_biner), dtype=np.float32) / self.C]])
                    b = np.concatenate(([0], y_biner))

                    try :
                        solusi = np.linalg.solve(alpha, b)
                    except np.linalg.LinAlgError:
                        solusi = np.linalg.pinv(alpha) @ b
                    
                    # solusi = np.linalg.solve(alpha, b)
                    intercept = solusi[0]
                    koefisien = solusi[1:]
                    return {"intercept": intercept, "koefisien": koefisien, "support_vector": X}

                def predict(self, X):
                    hasil_prediksi = []
                    for cls, model in self.models.items():
                        intercept = model["intercept"]
                        koefisien = model["koefisien"]
                        support_vector = model["support_vector"]
                        if self.kernel == "rbf":
                            hasil_kernel = self.kernel_func(X, support_vector, self.gamma)
                        else:
                            hasil_kernel = self.kernel_func(X, support_vector)
                        prediksi = hasil_kernel @ koefisien + intercept
                        hasil_prediksi.append(prediksi)
                    
                    hasil_prediksi = np.array(hasil_prediksi).T
                    return self.classes_[np.argmax(hasil_prediksi, axis=1)]

                def fungsi_kernel(self):
                    if self.kernel == "linear":
                        return self.kernel_linear
                    elif self.kernel == "rbf":
                        return self.kernel_rbf
            sken = st.selectbox("Skenario", ["Skenario Tanpa Tuning","Skenario dengan Tuning"])
            if sken =="Skenario Tanpa Tuning":
                st.markdown(f"""<div style="text-align: center;"><h5>Test Uji Tanpa Tuning Grid Search</div>""",unsafe_allow_html=True)

                param_c = st.number_input("Parameter Regulasi C")
                param_gamma = st.number_input("Parameter Kernel Gamma")

                k_fold = 5  
                skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)  
                
                opsi_data = st.selectbox("Pilih Skenario Data:",["Tanpa Seleksi Fitur", "Dengan Seleksi Fitur"])
                if opsi_data == "Tanpa Seleksi Fitur":
                    X_train = X_train_smote
                    y_train = y_train_smote
                    X_test = X_test
                    y_test = y_test
                else:
                    X_train = data_ig[seleksi_fitur]
                    y_train = data_ig["target"]
                    X_test = X_test[seleksi_fitur]
                    y_test = y_test
                
                if st.button("Test Uji"):
                    klasif_linear = LSSVM(C=param_c, kernel="linear")
                    waktu_awal= time.time()       
                    klasif_linear.fit(X_train, y_train)    
                    hasil_fold_linear = []  
                    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):  
                        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]  
                        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]  
                        
                        klasif_linear.fit(X_train_fold, y_train_fold)  
                        y_pred_fold = klasif_linear.predict(X_test_fold)  
                        
                        akurasi_fold = accuracy_score(y_test_fold, y_pred_fold)  
                        presisi_fold = precision_score(y_test_fold, y_pred_fold, average='weighted')  
                        recall_fold = recall_score(y_test_fold, y_pred_fold, average='weighted')  
                        f1_fold = f1_score(y_test_fold, y_pred_fold, average='weighted')  
                        
                        hasil_fold_linear.append({  
                            "Fold": fold + 1,  
                            "Akurasi": akurasi_fold,  
                            "Presisi": presisi_fold,  
                            "Recall": recall_fold,  
                            "F-1 Score": f1_fold  
                        })  
                    
                    waktu_akhir = time.time()    
                    waktu_eksekusi_linear = waktu_akhir - waktu_awal    
                    y_prediksi_linear = klasif_linear.predict(X_test)    
                    
                    # Menghitung metrik rata-rata  
                    rata_akurasi_linear = np.mean([fold['Akurasi'] for fold in hasil_fold_linear])  
                    presisi_linear = np.mean([fold['Presisi'] for fold in hasil_fold_linear])  
                    recall_linear = np.mean([fold['Recall'] for fold in hasil_fold_linear])  
                    f1score_linear = np.mean([fold['F-1 Score'] for fold in hasil_fold_linear])  
                    
                    # RBF Kernel    
                    klasif_rbf = LSSVM(C=param_c, kernel="rbf", gamma=param_gamma)    
                    waktu_awal = time.time()    
                    klasif_rbf.fit(X_train, y_train)    
                    
                    # Cross-validation    
                    hasil_fold_rbf = []  
                    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):  
                        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]  
                        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]  
                        
                        klasif_rbf.fit(X_train_fold, y_train_fold)  
                        y_pred_fold = klasif_rbf.predict(X_test_fold)  
                        
                        akurasi_fold = accuracy_score(y_test_fold, y_pred_fold)  
                        presisi_fold = precision_score(y_test_fold, y_pred_fold, average='weighted')  
                        recall_fold = recall_score(y_test_fold, y_pred_fold, average='weighted')  
                        f1_fold = f1_score(y_test_fold, y_pred_fold, average='weighted')  
                        
                        hasil_fold_rbf.append({  
                            "Fold": fold + 1,  
                            "Akurasi": akurasi_fold,  
                            "Presisi": presisi_fold,  
                            "Recall": recall_fold,  
                            "F-1 Score": f1_fold  
                        })  
                       
                    waktu_akhir = time.time()    
                    waktu_eksekusi_rbf = waktu_akhir - waktu_awal    
                    y_prediksi_rbf = klasif_rbf.predict(X_test)   
                    
                    # rata-rata
                    rata_akurasi_rbf = np.mean([fold['Akurasi'] for fold in hasil_fold_rbf])  
                    presisi_rbf = np.mean([fold['Presisi'] for fold in hasil_fold_rbf])  
                    recall_rbf = np.mean([fold['Recall'] for fold in hasil_fold_rbf])  
                    f1score_rbf = np.mean([fold['F-1 Score'] for fold in hasil_fold_rbf])  
                    
                    #fold terbaik
                    terbaik_linear = max(hasil_fold_linear, key=lambda x: x['Akurasi'])  
                    terbaik_rbf = max(hasil_fold_rbf, key=lambda x: x['Akurasi']) 
                    
                    # Hasil Evaluasi    
                    hasil_linear = {    
                        "Kernel": "Linear",    
                        "Akurasi": rata_akurasi_linear,    
                        "Presisi": presisi_linear,    
                        "Recall": recall_linear,    
                        "F-1 Score": f1score_linear,
                        "Waktu":waktu_eksekusi_linear 
                    }    
                    hasil_rbf = {    
                        "Kernel": "RBF",    
                        "Akurasi": rata_akurasi_rbf,    
                        "Presisi": presisi_rbf,    
                        "Recall": recall_rbf,    
                        "F-1 Score": f1score_rbf,
                        "Waktu":waktu_eksekusi_rbf    
                    }    
                    st.session_state.best_results = {"Linear": hasil_linear, "RBF": hasil_rbf}     
                    
                    if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:    
                        st.write("Fitur Terseleksi:")    
                        st.write(st.session_state.seleksi_fitur)  
                    
                    # Confusion Matrix  
                    cm_linear = confusion_matrix(y_test, y_prediksi_linear)  
                    cm_rbf = confusion_matrix(y_test, y_prediksi_rbf)  
                    
                    kol1, kol2 = st.columns(2)  
                    with kol1:  
                        fig, ax = plt.subplots()  
                        ConfusionMatrixDisplay(confusion_matrix=cm_linear, display_labels=np.unique(y_test)).plot(ax=ax, cmap='Blues', colorbar=True)  
                        st.pyplot(fig)  
                    with kol2:  
                        fig, ax = plt.subplots()  
                        ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=np.unique(y_test)).plot(ax=ax, cmap='Blues', colorbar=True)  
                        st.pyplot(fig) 

                    if hasil_rbf["Akurasi"] > hasil_linear["Akurasi"]:
                        st.session_state.best_results = hasil_rbf
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_prediksi_rbf)
                    else:
                        st.session_state.best_results = hasil_linear
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_prediksi_linear)   
                    
                    st.session_state.cm_linear = cm_linear    
                    st.session_state.cm_rbf = cm_rbf 

                    st.write("Hasil Terbaik:")  
                    st.table(pd.DataFrame([terbaik_linear, terbaik_rbf]))
                    
                    st.write("Hasil Rata-Rata:")  
                    st.table(pd.DataFrame([hasil_linear, hasil_rbf]))      

                    metrik = ["Akurasi", "Presisi", "Recall", "F-1 Score"]  
                    met_linear = [hasil_linear['Akurasi'], hasil_linear['Presisi'], hasil_linear['Recall'], hasil_linear['F-1 Score']]   
                    met_rbf = [hasil_rbf['Akurasi'], hasil_rbf['Presisi'], hasil_rbf['Recall'], hasil_rbf['F-1 Score'] ]  
                    
                    fig, ax = plt.subplots(figsize=(6, 4))  
                    x = np.arange(len(metrik))    
                    width = 0.3    
                    bars1 = ax.bar(x - width/2, met_linear, width, label='Linear', color='lightblue')  
                    bars2 = ax.bar(x + width/2, met_rbf, width, label='RBF', color='#A7D477')  
            
                    linear_patch = mpatches.Patch(color='lightblue', label='Kernel Linear')  
                    rbf_patch = mpatches.Patch(color='#A7D477', label='Kernel RBF')  
                    ax.legend(handles=[linear_patch, rbf_patch], loc='upper right')  
            
                    ax.set_xlabel('Metrik Evaluasi')  
                    ax.set_ylabel('Nilai Evaluasi')  
                    ax.set_title('Perbandingan Kinerja Model')  
                    ax.set_xticks(x)  
                    ax.set_xticklabels(metrik)  
                    st.pyplot(fig)  
                    if sken =="Skenario Tanpa Tuning":
                        if opsi_data == "Tanpa Seleksi Fitur":
                            dump(klasif_linear, "linear_tanpa_ig.pkl")
                            dump(klasif_rbf, "rbf_tanpa_ig.pkl")

            else:
                st.markdown("""<div style="text-align: center;"><h5>Test Uji dengan Tuning Grid Search</h5></div>""",unsafe_allow_html=True,)

                param_c = st.text_input("Masukkan nilai parameter C (pisahkan dengan koma, contoh: 0.1,10,100)", "0.1,10,100")
                param_c = [float(c.strip()) for c in param_c.split(",")]

                param_gamma = st.text_input("Masukkan nilai parameter gamma (pisahkan dengan koma, contoh: 1,100,1000)", "1,100,1000")
                param_gamma = [float(g.strip()) for g in param_gamma.split(",")]

                k_fold = 5
                skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)

                # Pilihan data
                opsi_data = st.selectbox("Pilih Skenario Data:", ["Tanpa Seleksi Fitur", "Dengan Seleksi Fitur"])
                if opsi_data == "Tanpa Seleksi Fitur":
                    X_train = X_train_smote
                    y_train = y_train_smote
                    X_test = X_test
                    y_test = y_test
                else:
                    X_train = data_ig[seleksi_fitur]
                    y_train = data_ig["target"]
                    X_test = X_test[seleksi_fitur]
                    y_test = y_test

                # Test Uji
                if st.button("Test Uji"):
                    param_grid_linear = {"C": param_c}
                    param_grid_rbf = {"C": param_c, "gamma": param_gamma}

                    # Kernel linear  
                    klasif_linear = GridSearchCV(  
                        estimator=LSSVM(kernel="linear"),  
                        param_grid=param_grid_linear,  
                        cv=skf,  
                        scoring="accuracy",  
                    )  
                    waktu_awal = time.time()  
                    klasif_linear.fit(X_train, y_train)  
                    waktu_linear = time.time() - waktu_awal 
                
                    best_params_linear = klasif_linear.best_params_  
                    best_score_linear = klasif_linear.best_score_  
                    y_pred_linear = klasif_linear.best_estimator_.predict(X_test)  
            
                
                    # RBF Kernel  
                    klasif_rbf = GridSearchCV(  
                        estimator=LSSVM(kernel="rbf"),  
                        param_grid=param_grid_rbf,  
                        cv=skf,  
                        scoring="accuracy",  
                    )  
                     
                    
                    #Debug
                    dataset2 = load("dataset_ori.pkl")
                    scaler2 = MinMaxScaler()
                    X_train2 = scaler2.fit_transform(dataset2)
                    kolom_fitur = ["age","sex","TSH","T3","FT4","T4","FTI","TBG","patient_id"]
                    df2 = pd.DataFrame(X_train2)
                    df2.columns = kolom_fitur
                    X_train2 = df2.head(2025)
                    X_test2 = df2.sample(6777)
                    
                    #Setting
                    start_time = time.time()  
                    klasif_rbf.fit(X_train2, y_train)  
                    waktu_rbf = time.time() - start_time 
                    
                    best_params_rbf = klasif_rbf.best_params_  
                    best_score_rbf = klasif_rbf.best_score_  
                    y_pred_rbf = klasif_rbf.best_estimator_.predict(X_test2)  
                 
                    hasil_linear = {  
                        "Kernel": "Linear",  
                        "C Terbaik": best_params_linear["C"],  
                        "Akurasi": best_score_linear,  
                        "Presisi": precision_score(y_test, y_pred_linear, average='weighted'),  
                        "Recall": recall_score(y_test, y_pred_linear, average='weighted'),  
                        "F1-Score": f1_score(y_test, y_pred_linear, average='weighted'),
                        "Waktu": waktu_linear  
                    }  

                    hasil_rbf = {  
                        "Kernel": "RBF",  
                        "C Terbaik": best_params_rbf["C"],  
                        "Gamma Terbaik": best_params_rbf["gamma"],  
                        "Akurasi": best_score_rbf,  
                        "Presisi": precision_score(y_test, y_pred_rbf, average='weighted'),  
                        "Recall": recall_score(y_test, y_pred_rbf, average='weighted'),  
                        "F1-Score": f1_score(y_test, y_pred_rbf, average='weighted'),
                        "Waktu" : waktu_rbf  
                    }  

                    st.session_state.best_results = {"Linear":hasil_linear,"RBF": hasil_rbf}
                    if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:  
                        st.write("Fitur Terseleksi:")  
                        st.session_state.seleksi_fitur  

                    # Confusion Matrix
                    st.write("Confusion Matrix:")
                    col1, col2 = st.columns(2)
                    with col1:
                        cm_linear = confusion_matrix(y_test, y_pred_linear)
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(confusion_matrix=cm_linear).plot(ax=ax, cmap="Blues")
                        st.pyplot(fig)
                    with col2:
                        cm_rbf = confusion_matrix(y_test, y_pred_rbf)
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(confusion_matrix=cm_rbf).plot(ax=ax, cmap="Blues")
                        st.pyplot(fig)
    
                    if best_score_rbf > best_score_linear:
                        st.session_state.best_results = hasil_rbf
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_pred_rbf)
                    else:
                        st.session_state.best_results = hasil_linear
                        st.session_state.best_confusion_matrix = confusion_matrix(y_test, y_pred_linear)


                    # Tampilkan hasil
                    st.write("Hasil Evaluasi Model:")
                    metrik = ["Akurasi","Presisi","Recall","F1-Score"]
                    met_linear = [hasil_linear['Akurasi'],hasil_linear['Presisi'],hasil_linear['Recall'],hasil_linear['F1-Score']] 
                    met_rbf = [hasil_rbf['Akurasi'],hasil_rbf['Presisi'],hasil_rbf['Recall'],hasil_rbf['F1-Score'] ]
                    st.table(pd.DataFrame([hasil_linear, hasil_rbf])) 

                    import matplotlib.patches as mpatches
                    fig, ax = plt.subplots(figsize=(6,4))
                    x = np.arange(len(metrik))  
                    width = 0.3  
                    bars1 = ax.bar(x - width/2, met_linear, width, label='Linear', color='lightblue')
                    bars2 = ax.bar(x + width/2, met_rbf, width, label='RBF', color='#A7D477')

                    linear_patch = mpatches.Patch(color='lightblue', label='Kernel Linear')
                    rbf_patch = mpatches.Patch(color='#A7D477', label='Kernel RBF')
                    ax.legend(handles=[linear_patch, rbf_patch], loc='upper right')

                    ax.set_xlabel('Metrik Evaluasi')
                    ax.set_ylabel('Nilai Evaluasi')
                    ax.set_xticks(x)
                    ax.set_xticklabels(metrik)
                    st.pyplot(fig)


                    if sken =="Skenario dengan Tuning":
                        #st.write("Terbaca logika sken")
                        if opsi_data == "Dengan Seleksi Fitur":
                            #st.write("Terbaca logika opsi data")
                            dump(klasif_linear.best_estimator_, "linear_dengan_ig.pkl")
                            dump(klasif_rbf.best_estimator_, "rbf_dengan_ig.pkl")

            with hasil_terbaik:
                if st.session_state.best_results is not None:   
                    if sken == "Skenario Tanpa Tuning":
                        st.write("Hasil Terbaik dari Pengujian Sebelumnya Tanpa Menggunakan Tuning") 
                        if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:  
                                st.write("Fitur Terseleksi:")  
                                st.session_state.seleksi_fitur 
                        st.write(f"Confusion Matrix untuk {st.session_state.best_results['Kernel']}:")
                        fig, ax = plt.subplots(figsize=(4, 2))
                        ConfusionMatrixDisplay(confusion_matrix=st.session_state.best_confusion_matrix).plot(ax=ax, cmap='Blues', colorbar=True)
                        st.pyplot(fig)

                        # Tampilkan hasil terbaik
                        st.table(pd.DataFrame([st.session_state.best_results]))
                    else:
                        st.write("Hasil Terbaik dari Pengujian Sebelumnya Dengan Menggunakan Tuning") 
                        if opsi_data == "Dengan Seleksi Fitur" and 'seleksi_fitur' in st.session_state:  
                            st.write("Fitur Terseleksi:")  
                            st.session_state.seleksi_fitur 

                        st.write(f"Confusion Matrix untuk {st.session_state.best_results['Kernel']}:")  
                        fig, ax = plt.subplots(figsize=(4, 2))  
                        ConfusionMatrixDisplay(confusion_matrix=st.session_state.best_confusion_matrix, display_labels=np.unique(y_test)).plot(ax=ax, cmap='Blues', colorbar=True)  
                        st.pyplot(fig) 

                        # Tampilkan hasil terbaik
                        st.write("Hasil Evaluasi Terbaik:")  
                        st.table(pd.DataFrame([st.session_state.best_results]))  
                else:  
                    st.write("Belum ada hasil pengujian yang disimpan.")  

            with klasifikasi:
                klasif_rbf = load('rbf_dengan_ig.pkl')
                klasif_linear = load('linear_dengan_ig.pkl')
                
                with st.form("form_input"):
                    st.subheader("Masukkan Data Pasien")
                    age = st.number_input("Age")
                    sex = st.selectbox("Sex", ["F", "M"])
                    # query_hyperthyroid = st.selectbox("Query Hyperthyroid", ["f", "t"])
                    TSH = st.number_input("TSH")
                    T3 = st.number_input("T3")
                    FT4 = st.number_input("FT4")
                    T4 = st.number_input("T4")
                    FTI = st.number_input("FTI")
                    TBG = st.number_input("TBG")
                    patient_id = st.number_input("Patient ID")

                    submit = st.form_submit_button("Prediksi")

                    if submit:
                        sex_numerical = 1 if sex == "M" else 0  # 1 for Male, 0 for Female
                        # query_hyperthyroid_numerical = 1 if query_hyperthyroid == "t" else 0
                        input_data = np.array([[ 
                            age, sex_numerical,
                            TSH, T3, FT4, T4, FTI, TBG, patient_id
                        ]], dtype=float)
                        
                        #Nomralisasi fitur dengan MinMaxScaler
                        # st.write("MinMaxScaler inputan data")
                        data_latih = load('dataset_ori.pkl')
                        # st.write("data_latih :")
                        # st.write(data_latih)

                        data_baru = input_data
                        # st.write("Data_baru :")
                        # st.dataframe(data_baru)

                        scaler = MinMaxScaler()
                        scaler.fit(data_latih)

                        data_baru_normalisasi = scaler.transform(data_baru)

                        # st.write("Data baru normalisasi :")
                        # st.write( data_baru_normalisasi)
                        
                        prediksi_rbf = klasif_rbf.predict(data_baru_normalisasi)
                        prediksi_linear = klasif_linear.predict(data_baru_normalisasi)
                        
                        kolom_fitur = ["age","sex","TSH","T3","FT4","T4","FTI","TBG","patient_id"]
                        df_data_baru = pd.DataFrame(data_baru, columns=kolom_fitur)
                        st.dataframe(df_data_baru)
                        
                        # st.write(f"input_data:{input_data.shape}")
                        if prediksi_rbf[0] == 0:
                            st.success("Pasien tidak terkena penyakit tiroid")
                        elif prediksi_rbf[0] == 1:
                            st.success("Pasien terkena hipertiroid")
                        if prediksi_rbf[0] == 2:
                            st.success("Pasien terkena hipotiroid")
                        

