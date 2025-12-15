# RNN_p

**RNN_p** is a C++ library for training RNN(p) models, as described in the paper 
"RNN(p) for Power Consumption Forecasting". This library provides a simple and flexible pipeline for time-series 
forecasting, leveraging modular design for linear algebra operations, data handling, model training, and preprocessing.

---

## 📚 Authors

<!-- - **[Pietro Manzoni](Author-1-Profile-Link)** -->
- Pietro Manzoni, University of Edinburgh, UK
- Roberto Baviera, Politecnico di Milano, Italy

<!--For detailed references, please see the paper: ["RNN(p) for Power Consumption Forecasting"](Link-to-Paper).  -->

For detailed references, please see the paper: "RNN(p) for Power Consumption Forecasting".

---

## 📁 Project Structure

```text
RNNSimple/
├── CMakeLists.txt              # Build configuration
├── README.md                   # Project documentation
├── config.txt                  # Configuration file
├── data/                       # Input datasets (.csv files)
├── results/                    # Outputs and logs
├── src/                        # All source code
│   ├── main.cpp                # Main code
│   ├── dataframe/              # DataFrame-related code
│   │   ├── DataFrame.h
│   │   └── DataFrame.cpp
│   ├── linalg/                 # Linear algebra operations
│   │   ├── AlgebraicOperations.h
│   │   ├── AlgebraicOperations.cpp
│   │   ├── Matrix.h
│   │   ├── Matrix.cpp
│   │   ├── Vector.h
│   │   └── Vector.cpp
│   ├── models/                 # Models (Linear and Neural)
│   │   ├── LinearModel.h
│   │   ├── LinearModel.cpp
│   │   ├── Recurrent.h
│   │   └── Recurrent.cpp
│   ├── preprocessing/          # Preprocessing code (scaler)
│   │   ├── Scaler.h
│   │   └── Scaler.cpp
│   └── utils/                  # Utility functions
│       ├── utils.h
│       └── utils.cpp
```

---

## ⚙️ Configuration

All runtime parameters are stored in `config.txt`. Update this file to change dataset paths, model settings, 
or hyperparameters.

Make sure your application loads this file properly at runtime.

---

# 📈 Results
All output (e.g., logs, predictions, performance evaluation) are written to the `results/` directory

---

## 🛠️ Build Instructions

To build and run the project:

### 🔹 On Linux / macOS

```bash
mkdir build && cd build
cmake ..
make
./RNN_p
```

### 🔸 On Windows (Visual Studio)
1. Open the x64 Native Tools Command Prompt for VS

2. Run:

```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
.\Release\RNN_p.exe
```
> 💡 If you're using a different Visual Studio version, adjust the generator accordingly.

---

# 📄 Dependencies
* C++20 standard
* CMake ≥ 3.21
* No external libraries required — runs on standard C++.

---

# ⚖️ License 

This project is licensed under the GNU General Public License (GPL-3.0).

### What Does the GPL-3.0 License Mean?

- You are free to **use, modify, and distribute** the project as long as:
    - You **share the source code** (or provide a way to access it).
    - You **distribute your modifications** under the same GPL-3.0 license.
    - Any **derivative works** (i.e., code based on this project) must also be licensed under the GPL-3.0.

For more information on the GPL-3.0 License, visit: [GNU.org GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html).
