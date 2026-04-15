# # # # src/data_loader.py
# # # import pandas as pd
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# # # # Import the configuration settings
# # # from src import config

# # # def load_and_preprocess_data():
# # #     """
# # #     This function loads the dataset, performs cleaning and preprocessing,
# # #     and splits it into training and testing sets.
    
# # #     Returns:
# # #         Tuple: A tuple containing X_train, X_test, y_train, y_test,
# # #                the fitted scaler, and the fitted label_encoder.
# # #     """
# # #     # Load the dataset using the path from the config file
# # #     print("Loading data...")
# # #     df = pd.read_csv(config.DATA_PATH)
    
# # #     # --- Data Cleaning ---
# # #     # Drop duplicate rows to prevent data leakage
# # #     df.drop_duplicates(inplace=True)
    
# # #     # Separate features (X) and the target label (y)
# # #     X = df.drop('label', axis=1)
# # #     y = df['label']

# # #     # --- Label Engineering ---
# # #     # Group rare attack classes into a single 'Other_Attack' category
# # #     print("Grouping rare attack classes...")
# # #     label_counts = y.value_counts()
# # #     rare_labels = label_counts[label_counts < config.RARE_LABEL_THRESHOLD].index
# # #     y_grouped = y.replace(rare_labels, 'Other_Attack')
    
# # #     # --- Feature Scaling ---
# # #     # Scale numerical features to a range of [0, 1] for the autoencoder
# # #     print("Scaling features...")
# # #     scaler = MinMaxScaler()
# # #     X_scaled = scaler.fit_transform(X)

# # #     # --- Label Encoding ---
# # #     # Convert string labels into integers
# # #     print("Encoding labels...")
# # #     label_encoder = LabelEncoder()
# # #     y_encoded = label_encoder.fit_transform(y_grouped)

# # #     # --- Data Splitting ---
# # #     # Split the data into training and testing sets (80/20 split)
# # #     # 'stratify=y_encoded' ensures the label distribution is the same in both sets
# # #     print("Splitting data into training and testing sets...")
# # #     X_train, X_test, y_train, y_test = train_test_split(
# # #         X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# # #     )

# # #     print("Data loading and preprocessing complete.")
    
# # #     # Return all the necessary components for training and evaluation
# # #     return X_train, X_test, y_train, y_test, scaler, label_encoder




# # # src/data_loader.py
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# # from imblearn.over_sampling import SMOTE # <-- ADD THIS IMPORT
# # import numpy as np

# # # Import the configuration settings
# # from src import config

# # def load_and_preprocess_data():
# #     """
# #     This function loads the dataset, performs cleaning, preprocessing,
# #     applies SMOTE to the training data, and splits it.
# #     """
# #     print("Loading data...")
# #     df = pd.read_csv(config.DATA_PATH)
    
# #     # --- Data Cleaning ---
# #     df.drop_duplicates(inplace=True)
    
# #     X = df.drop('label', axis=1)
# #     y = df['label']

# #     # --- Label Engineering ---
# #     print("Grouping rare attack classes...")
# #     label_counts = y.value_counts()
# #     rare_labels = label_counts[label_counts < config.RARE_LABEL_THRESHOLD].index
# #     y_grouped = y.replace(rare_labels, 'Other_Attack')
    
# #     # --- Feature Scaling ---
# #     print("Scaling features...")
# #     scaler = MinMaxScaler()
# #     X_scaled = scaler.fit_transform(X)

# #     # --- Label Encoding ---
# #     print("Encoding labels...")
# #     label_encoder = LabelEncoder()
# #     y_encoded = label_encoder.fit_transform(y_grouped)

# #     # --- Data Splitting ---
# #     print("Splitting data into training and testing sets...")
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# #     )
    
# #     # ===================================================================
# #     #   NEW: Apply SMOTE to the training data ONLY
# #     # ===================================================================
# #     print("\nApplying SMOTE to balance the training data...")
# #     print(f"Shape of training data BEFORE SMOTE: {X_train.shape}")
    
# #     # We use k_neighbors=3 because some rare classes might have fewer than 5 samples
# #     # which is the default for SMOTE.
# #     smote = SMOTE(random_state=42, k_neighbors=3)
# #     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
# #     print(f"Shape of training data AFTER SMOTE: {X_train_resampled.shape}")
# #     print(f"Number of training samples increased by: {X_train_resampled.shape[0] - X_train.shape[0]}")
    
# #     print("\nData loading and preprocessing complete.")
    
# #     # Return the resampled training data along with the original test data
# #     return X_train_resampled, X_test, y_train_resampled, y_test, scaler, label_encoder





# # src/data_loader.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# import numpy as np

# # Import the configuration settings
# from src import config

# def load_and_preprocess_data():
#     """
#     This version simplifies the problem by focusing on major attack categories
#     and grouping all other attacks into a single 'Other_Attack' class.
#     SMOTE is removed to prioritize accuracy on dominant classes.
#     """
#     print("Loading data...")
#     df = pd.read_csv(config.DATA_PATH)
    
#     # --- Data Cleaning ---
#     df.drop_duplicates(inplace=True)
    
#     X = df.drop('label', axis=1)
#     y = df['label']

#     # ===================================================================
#     #   NEW: Smart Label Grouping for High Accuracy
#     # ===================================================================
#     print("Grouping labels into major categories...")

#     # Define the major attack families we want to identify specifically.
#     # These are the high-volume or critical attack types.
#     major_attacks = [
#         'BenignTraffic',
#         'DDoS-ICMP_Flood',
#         'DDoS-PSHACK_Flood',
#         'DDoS-RSTFINFlood',
#         'DDoS-SYN_Flood',
#         'DDoS-SynonymousIP_Flood',
#         'DDoS-TCP_Flood',
#         'DDoS-UDP_Flood',
#         'DoS-SYN_Flood',
#         'DoS-TCP_Flood',
#         'DoS-UDP_Flood',
#         'Mirai-greeth_flood',
#         'Mirai-greip_flood',
#         'Mirai-udpplain'
#     ]

#     # Any label NOT in our major_attacks list will be renamed to 'Other_Attack'.
#     y_simplified = y.apply(lambda x: x if x in major_attacks else 'Other_Attack')

#     print("New simplified label distribution:")
#     print(y_simplified.value_counts())
    
#     # --- Feature Scaling ---
#     print("\nScaling features...")
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     # --- Label Encoding ---
#     print("Encoding labels...")
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y_simplified)

#     # --- Data Splitting ---
#     print("Splitting data into training and testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )
    
#     print("\nData loading and preprocessing complete.")
    
#     # We return the original training data (no SMOTE) and the simplified labels
#     return X_train, X_test, y_train, y_test, scaler, label_encoder





# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Switched to StandardScaler
import numpy as np
from src import config

def load_and_preprocess_data():
    """
    This version uses a smart taxonomy to group detailed labels into 7 major categories,
    which is the key to achieving high, stable accuracy.
    """
    print("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    df.drop_duplicates(inplace=True)
    
    # This taxonomy map is the core feature for simplifying the problem
    taxonomy_map = {
        "Flood Attacks": [
            "DDoS-ICMP_Flood", "DDoS-TCP_Flood", "DDoS-UDP_Flood", "DoS-TCP_Flood", 
            "DoS-UDP_Flood", "DoS-SYN_Flood", "DDoS-PSHACK_Flood", "DDoS-RSTFINFlood", 
            "DoS-HTTP_Flood", "DDoS-SYN_Flood", "DDoS-SynonymousIP_Flood", "DDoS-HTTP_Flood",
            "DDoS-ACK_Fragmentation", "DDoS-ICMP_Fragmentation", "DDoS-UDP_Fragmentation",
            "DDoS-SlowLoris"
        ],
        "Botnet/Mirai Attacks": [
            "Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood"
        ],
        "Reconnaissance": [
            "Recon-HostDiscovery", "Recon-OSScan", "Recon-PortScan", "Recon-PingSweep",
            "VulnerabilityScan"
        ],
        "Spoofing / MITM": [
            "DNS_Spoofing", "MITM-ArpSpoofing"
        ],
        "Injection Attacks": [
            "SqlInjection", "CommandInjection", "XSS"
        ],
        "Backdoors & Exploits": [
            "Backdoor_Malware", "Uploading_Attack", "BrowserHijacking", "DictionaryBruteForce"
        ],
        "Benign": [
            "BenignTraffic"
        ]
    }
    
    label_to_taxonomy = {label: category for category, labels in taxonomy_map.items() for label in labels}
    df['taxonomy_label'] = df['label'].map(label_to_taxonomy)
    
    # Drop rows where the label couldn't be mapped (if any)
    df_clean = df.dropna(subset=['taxonomy_label']).copy()
    
    print("New simplified label distribution:")
    print(df_clean['taxonomy_label'].value_counts())
    
    X = df_clean.drop(columns=['label', 'taxonomy_label'])
    y_simplified = df_clean['taxonomy_label']
    
    print("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_simplified)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\nData loading and preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler, label_encoder, X.columns