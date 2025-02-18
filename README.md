# Reinforcement Learning in AI 🚀  

This repository contains **various reinforcement learning (RL) projects** that demonstrate the application of **tabular learning, deep reinforcement learning, and Monte Carlo Tree Search (MCTS)**. Each project focuses on solving different AI-related problems using advanced learning techniques.

## 📌 Overview
This repository is divided into several **independent projects**, each implementing a different AI-based game agent using **machine learning and reinforcement learning techniques**.

| Project | Algorithm | Description |
|---------|------------|-------------|
| **Passenger Satisfaction Prediction** | **Neural Networks (MLP)** | Predicts airline passenger satisfaction using **deep learning** models. |
| **Connect Four AI** | **Monte Carlo Tree Search (MCTS)** | Implements **an AI opponent** for Connect Four using MCTS-based decision-making. |
| **Jacky Card Game** | **Tabular Q-Learning & Deep RL** | Implements **Q-learning** for a Blackjack-inspired game, followed by **Deep Q-Networks (DQN)**. |
| **Snort Board Game AI** | **PUCT (Predictor Upper Confidence Tree)** | AI player using **Neural Network-guided Monte Carlo Tree Search (MCTS)** with **self-play training**. |

Each project follows a structured approach:
1. **Game/Problem Implementation** – Define rules and logic.
2. **Algorithmic Implementation** – Apply RL or ML techniques.
3. **Training & Optimization** – Train AI using simulated environments.
4. **Evaluation & Performance Metrics** – Analyze the model’s accuracy and efficiency.

---

## 📂 Project Structure
📂 Reinforcement_Learning_In_AI
 
   ├── 📂 PassengerSatisfactionProject    # Neural Network for satisfaction prediction

   ├── 📂 ConnectFourProject              # MCTS-based AI for Connect Four
 
   ├── 📂 TabularLearningAndDeepRL        # Monte Carlo & Deep RL for Jacky game
 
   ├── 📂 FinalProject-Snort              # PUCT-based AI for Snort board game
 
   ├── README.md                          # Project documentation



---

## 📌 Project Details
### **1️⃣ Passenger Satisfaction Prediction**
**Algorithm Used:** Multi-Layer Perceptron (MLP) Neural Network  

📌 **Description:**  
- A **classification model** that predicts **airline passenger satisfaction** based on customer feedback.  
- Uses a **deep neural network** trained on structured **Kaggle datasets**.  

📌 **Key Features:**
- Data preprocessing: Handling missing values and encoding categorical features.
- **Neural Network architecture:** Fully connected layers with ReLU activations.
- Optimized with **Adam optimizer** and **cross-entropy loss**.

**How to run:**  
```bash
python ex2.py  # Train and evaluate the model
```

---

### 2️⃣ Connect Four AI (MCTS)

### **Algorithm Used:** Monte Carlo Tree Search (MCTS)

📌 **Description:**
- Implements an **AI player for Connect Four**, allowing a **human vs. AI** match.
- Uses **MCTS** to explore possible moves and simulate thousands of game states.

📌 **Key Features:**
- **Game logic implementation:** `ConnectFour.py`
- **MCTS AI player:** `MCTSPlayer.py`
- **Tree search exploration** using **Upper Confidence Bound (UCB1)**.

### **How to Run:**
```bash
python ConnectFour.py  # Play against the AI
```

---

### 3️⃣ Jacky Game AI (Tabular Learning & Deep RL)

### **Algorithm Used:** Tabular Q-Learning, Deep Q-Networks (DQN)

📌 **Description:**
-  Implements Monte Carlo Learning for a simplified Blackjack variant (Jacky).
-  Extends the model to Deep Q-Networks (DQN) for learning optimal policies with randomized rewards.
   
📌 **Key Features:**
- **Tabular Q-Learning:** Uses a Q-table for state-action estimation.
- **DQN model:** Uses TensorFlow/Keras to approximate Q-values.

### **How to Run:**
```bash
python Jacky.py  # Train and evaluate the AI
```
    
---


### 4️⃣ Snort Game AI (PUCT & Self-Play)

### **Algorithm Used:** Predictor Upper Confidence Tree (PUCT) + Deep Neural Networks

📌 **Description:**
-  Implements PUCT-based reinforcement learning for the Snort board game.
-  AI trains through self-play and improves using a neural network for policy evaluation.
   
📌 **Key Features:**
- **Stage 1:** Implements Snort game logic (SnortGame.py).
- **Stage 2:** Uses PUCT algorithm to explore optimal strategies.
- **Deep Learning:** Trains a policy-value network to guide the AI.

### **How to Run:**
```bash
python Ex8-Snort/main.py  # Train and play against the AI
```

---

## 📌 Installation & Setup

1. Clone the Repository

```bash
git clone https://github.com/coral555/Reinforcement_Learning_In_AI.git
cd Reinforcement_Learning_In_AI
```

2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Run a Specific Project

Navigate to the relevant folder and execute the script:

```bash
python project_script.py
```

For example, to play against the Connect Four AI:

```bash
python ConnectFour.py
```

---

## 👨‍💻 Author

This repository is maintained by Coral Bahofrker, a software engineer specializing in Artificial Intelligence, Reinforcement Learning, and Neural Networks.

If you have any questions or suggestions, feel free to reach out or open an issue. 

📧Contact: coralb2001@gmail.com

✅ If you find this repository useful, don't forget to ⭐ star the repo! 😊
