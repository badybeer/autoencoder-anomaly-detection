# Autoencoder-anomaly-detection
System based on autoencoder to detect anomalies in network traffic

# Program Function

The program is designed for the detection of anomalies in network traffic using a traditional autoencoder. Its primary function is the prediction of the `bidirectional_packets` attribute, based on which it classifies anomalous network behavior. The attribute `bidirectional_first_seen_ms` is used for timestamping events.

# Program Installation

Files with the `.ipynb` extension are opened in `JupyterLab` or another program that supports `.ipynb` files. Before running the program, it is necessary to install the required packages in the `JupyterLab` environment (or your chosen alternative). The list of packages can be found in the `requirements.txt` file or in the executable `installation_requirements.ipynb` for direct installation.

## Technical Requirements

The model training was conducted on servers with the following hardware specifications:

| Component | Specification              |
|-----------|----------------------------|
| Server    | HP ProLiant DL380 Gen9     |
| CPU       | 2x Intel Xeon E5-2697A v4  |
| GPU       | Nvidia RTX A4000 16GB      |
| RAM       | 64GB DDR4 2400MHz          |

## Software Requirements

To run the program, you need to have the JupyterLab environment installed or another alternative that supports `.ipynb` files. It is also necessary to install the list of packages mentioned in the `requirements.txt` file or through the executable file `installation_requirements.ipynb` for direct installation.

## Program Structure Description

The source code with the user interface is located in the file `Anomaly_Detection.ipynb`. This is a Jupyter Notebook file, which is characterized by dividing the code into blocks. Each block represents a partial functionality of the entire task.

## Description of Messages for System Programmers

If a block of source code has an expected output, such as data shape, data type, architecture image, model summarization, performance evaluation or errors, evaluation of metrics, or a contingency table, this output will be displayed in the console under the source code block after execution.

# Description of Input, Output, and Working Files

These files are located in a folder named `Datasets` in the file `27072022.csv` (not uploaded to GitHub due to dataset size). Using the code in the file `Dataset_Processing.ipynb`, we subsequently divide the dataset size into smaller datasets, which are stored in a folder named `Datasets/EditDatasets` in files named from `split_1.csv` to `split_100.csv`. The output of the program is a trained neural network based on autoencoders, capable of predicting the `bidirectional_packets` attribute and detecting anomalous behavior in network traffic based on this attribute.

# Minimum Requirements
To operate the software necessary to run JupyterLab, which is used in experiments and project work, you must have:
- A computer with an operating system such as Windows, Linux, or macOS, meeting the minimum requirements to run JupyterLab.

## Minimum System Requirements for JupyterLab
For effective use of JupyterLab, the following minimum system requirements must be met:
- **Operating System**: Support for current versions of Windows, macOS, and Linux.
- **Processor**: Modern processor, at least 1 GHz or faster.
- **RAM**: Minimum 1 GB, recommended 4 GB or more for working with larger data files or more complex calculations.
- **Data Storage**: At least 1 GB of free disk space for JupyterLab itself and additional space for your projects and data.
- **Graphics Card**: No specific graphics card is needed for regular use. However, having a powerful GPU can be beneficial for working with graphs, data visualization, or computational graphics.
- **Web Browser**: Latest version of Google Chrome, Mozilla Firefox, Safari, or Microsoft Edge.
- **Python**: Python 3.6 or newer is required to install JupyterLab.

# Program Function
The program is designed to detect anomalies in network traffic using an autoencoder, a specific type of neural network. Its goal is to identify atypical patterns in the `bidirectional_packets` attribute, which may indicate potential security threats, such as DDoS attacks. The program is implemented in Python using libraries for deep learning and data analysis.

# Solution Analysis
The solution focuses on the efficient identification of anomalies using an autoencoder, which is trained on normal data points. Anomalies are then detected as inputs that exhibit high reconstruction error. The process involves data preprocessing, model training, and evaluation on a test set.

# Program Description
The program is organized into blocks in a `Jupyter Notebook`, allowing for interactive code execution and visualization of results. These files are located in a folder named `Datasets` and are saved in the file `27072022.csv` (not uploaded to GitHub due to dataset size). Using the code in the file `Dataset_Processing.ipynb`, we subsequently divide the dataset size into smaller datasets, which are stored in a folder named `Datasets/EditDatasets` in files named from `split_1.csv` to `split_100.csv`. The source code with the user interface is located in the file `Anomaly_Detection.ipynb`. The output of the program is a trained neural network based on autoencoders, capable of predicting the attribute `bidirectional_packets` and detecting anomalous behavior in network traffic based on this attribute.

## Solution Description
The solution uses a series of steps for data preprocessing, designing the autoencoder, training it, and testing. This approach allows the model to learn the representation of normal data points, based on which it can detect anomalies.

## Description of Algorithms and Data Structures, Global Variables
The main script `Anomaly_Detection.ipynb` is composed of small blocks. Due to the extensive code for data visualization, we will focus only on the most important blocks, which we will now describe:

- **Block 1**: Imports necessary libraries for data manipulation, visualization, preprocessing, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,
    ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
```
Block 2: Loading input data from a CSV file.

data = pd.read_csv("Datasets/EditDatasets/split_1.csv")

Block 3: Selecting required columns.

selected_columns = ['bidirectional_packets', 'bidirectional_first_seen_ms']
data_selected = data[selected_columns]

Block 4: Converts a column containing timestamps in Unix timestamp format (in milliseconds) to regular timestamps and saves them in a new column 'timestamp' in the dataframe.

data['timestamp'] = pd.to_datetime(data['bidirectional_first_seen_ms'], unit='ms')

Here's how you can convert the LaTeX content to a README.md format suitable for GitHub. I've translated each section to Markdown, ensuring readability and maintaining the structure for clarity:

markdown

# Minimum Requirements
To operate the software necessary to run JupyterLab, which is used in experiments and project work, you must have:
- A computer with an operating system such as Windows, Linux, or macOS, meeting the minimum requirements to run JupyterLab.

## Minimum System Requirements for JupyterLab
For effective use of JupyterLab, the following minimum system requirements must be met:
- **Operating System**: Support for current versions of Windows, macOS, and Linux.
- **Processor**: Modern processor, at least 1 GHz or faster.
- **RAM**: Minimum 1 GB, recommended 4 GB or more for working with larger data files or more complex calculations.
- **Data Storage**: At least 1 GB of free disk space for JupyterLab itself and additional space for your projects and data.
- **Graphics Card**: No specific graphics card is needed for regular use. However, having a powerful GPU can be beneficial for working with graphs, data visualization, or computational graphics.
- **Web Browser**: Latest version of Google Chrome, Mozilla Firefox, Safari, or Microsoft Edge.
- **Python**: Python 3.6 or newer is required to install JupyterLab.

# Program Function
The program is designed to detect anomalies in network traffic using an autoencoder, a specific type of neural network. Its goal is to identify atypical patterns in the `bidirectional_packets` attribute, which may indicate potential security threats, such as DDoS attacks. The program is implemented in Python using libraries for deep learning and data analysis.

# Solution Analysis
The solution focuses on the efficient identification of anomalies using an autoencoder, which is trained on normal data points. Anomalies are then detected as inputs that exhibit high reconstruction error. The process involves data preprocessing, model training, and evaluation on a test set.

# Program Description
The program is organized into blocks in a `Jupyter Notebook`, allowing for interactive code execution and visualization of results. These files are located in a folder named `Datasets` and are saved in the file `27072022.csv`. Using the code in the file `Dataset_Processing.ipynb`, we subsequently divide the dataset size into smaller datasets, which are stored in a folder named `Datasets/EditDatasets` in files named from `split_1.csv` to `split_100.csv`. The source code with the user interface is located in the file `Anomaly_Detection.ipynb`. The output of the program is a trained neural network based on autoencoders, capable of predicting the attribute `bidirectional_packets` and detecting anomalous behavior in network traffic based on this attribute.

## Solution Description
The solution uses a series of steps for data preprocessing, designing the autoencoder, training it, and testing. This approach allows the model to learn the representation of normal data points, based on which it can detect anomalies.

## Description of Algorithms and Data Structures, Global Variables
The main script `Anomaly_Detection.ipynb` is composed of small blocks. Due to the extensive code for data visualization, we will focus only on the most important blocks, which we will now describe:

- **Block 1**: Imports necessary libraries for data manipulation, visualization, preprocessing, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,
    ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
```

- **Block 2**: Loading input data from a CSV file.

```python
data = pd.read_csv("Datasets/EditDatasets/split_1.csv")
```

- **Block 3**: Selecting required columns.

```python
selected_columns = ['bidirectional_packets', 'bidirectional_first_seen_ms']
data_selected = data[selected_columns]
```

- **Block 4**: Converts a column containing timestamps in Unix timestamp format (in milliseconds) to regular timestamps and saves them in a new column 'timestamp' in the dataframe.

```python
data['timestamp'] = pd.to_datetime(data['bidirectional_first_seen_ms'], unit='ms')
```

- **Block 5**:
  Selects the bidirectional_packets column from the dataframe and converts it to an array of values.
    - Normalizes the data using MinMaxScaler.
    - Splits the data into training and test sets.

```python
packets_data = data[['bidirectional_packets']].values

scaler = MinMaxScaler()
packets_normalized = scaler.fit_transform(packets_data)

X_train, X_test = train_test_split(packets_normalized, test_size=0.3, random_state=42, shuffle=False)
X_train.shape, X_test.shape
```

- **Block 6**: Defines the architecture of the autoencoder.

```python
input_layer = Input(shape=(1,))
encoded = Dense(8, activation='relu', activity_regularizer=l2(1e-5))(input_layer)
encoded = Dropout(0.1)(encoded)
encoded = Dense(4, activation='relu', activity_regularizer=l2(1e-5))(encoded)
encoded = Dropout(0.1)(encoded)
encoded = Dense(2, activation='relu', activity_regularizer=l2(1e-5))(encoded)

decoded = Dense(4, activation='relu')(encoded)
decoded = Dropout(0.1)(decoded)
decoded = Dense(8, activation='relu')(decoded)
decoded = Dropout(0.1)(decoded)
decoded = Dense(1, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stopping, reduce_lr]
)
```

- **Block 7**:
  - Predicts the test set.
  - Calculates the reconstruction error.

```python
X_test_pred = autoencoder.predict(X_test)

reconstruction_error = np.mean(np.square(X_test_pred - X_test), axis=1)
```

- **Block 8**: Calculates the threshold.

```python
threshold = np.percentile(reconstruction_error, 95)
```

- **Block 9**: Identifies anomalies.

```python
anomalies = X_test[reconstruction_error > threshold]
```

The blocks not described here are auxiliary and will have comments in the code explaining their functionality.

# List of Source Files

- **Primary**:
    - Anomaly_Detection.ipynb - main script containing the autoencoder for anomaly detection.
    - Dataset_Processing.ipynb - script for dividing the main dataset into smaller ones.
    - installation_requirements.ipynb - script for installing necessary libraries.
    - requirements.txt - list of required packages.
    - /Datasets/27072022.csv - main dataset. (not uploaded to GitHub due to dataset size)
    - /Datasets/EditDatasets/from split_1.csv to split_100.csv - edit main dataset.
- **Secondary**:
    - Visualisation_Dataset.ipynb - auxiliary visualization of the main dataset.
    - Visualization_Top10_Protocol.ipynb - auxiliary visualization of protocols.
