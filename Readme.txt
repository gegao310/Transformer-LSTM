
#Overview
This repository contains hybrid neural network architectures combining Transformer and LSTM components for EMG signal processing. Three distinct model variants are implemented for different temporal modeling scenarios.

#Model Architecture

├── Network/

│   ├── transformer_and_lstm_model.py       # Core network code

│   │   ├── TransformerEMG()                # Transformer model with encoder and self-attention mechanisms

│   │   ├── MyModule()                      # Transformer-LSTM hybrid model without autoregression

│   │   └── TransformerLSTM_autoregression_model()  # Transformer-LSTM model with autoregression

├── across.py                               # Cross-subject plotting

├── selection.py                            # Channel selection plotting

└── README.md

##1. TransformerEMG (TransformerEMG class)
###‌Core Components‌:
Input linear projection layer

Transformer encoder stack

Sequence reduction module

###‌Key Features‌:
Self-attention mechanism for global temporal dependencies

Position-aware encoding

Sequence dimension reduction

##2. Non-autoregressive Hybrid (MyModule class)

###‌Architecture‌:
graph TD

  A[Input EMG] --> B(24-sample Segments)
  
  B --> C[TransformerEMG]
  
  C --> D{Concatenate}
  
  D --> E[LSTM]
  
  E --> F[Final Prediction]
  
###‌Characteristics‌:
Parallel temporal segment processing

Context aggregation via LSTM

Fixed-window segmentation

##3.Autoregressive Hybrid (Transformer_and_Lstm_autoregression_model class)

###‌Architecture‌:
graph TD
  A[Input EMG] --> B[TransformerEMG]
  
  B --> C(24-step Sequence)
  
  C --> D[LSTM Cell]
  
  D --> E{Prediction Step}
  
  E -->|Loop| C
  
###‌Characteristics‌:
Iterative prediction generation

Sequential dependency modeling

Dynamic output conditioning

#Training Example

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()

for epoch in range(100):

    for batch in dataloader:
    
        inputs, targets = batch
        
        outputs = model(inputs) 