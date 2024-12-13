# ML-Based Cloud Resource Optimization

## **Overview**
Efficient resource allocation in cloud computing environments is a critical challenge. This project leverages machine learning (ML) models and cloud-native technologies to dynamically optimize the allocation of Virtual Machines (VMs) based on predicted CPU workloads. The system integrates predictive ML models with a scalable backend and frontend architecture, deployed on **Google Cloud Platform (GCP)**.

## **Key Features**
- **Predictive ML Models**: Implements Linear Regression, GRU, and Bidirectional LSTM for workload prediction.
- **Dynamic Resource Allocation**: Simulates real-time VM allocation using SimPy, adjusting to predicted workloads.
- **Cloud-Native Deployment**: Utilizes GCP services like Cloud Run, Firebase Hosting, and Cloud Storage for scalability and accessibility.
- **Interactive Dashboard**: A React-based frontend visualizes predicted workloads and allocation performance.

---

## **Architecture**
### **System Components**
1. **Frontend**:
   - Built with React and Chart.js for visualization.
   - Hosted on **Firebase Hosting**.
2. **Backend**:
   - Developed with Node.js and Express.
   - Exposes APIs for workload predictions and VM scheduling.
   - Deployed on **Google Cloud Run**.
3. **Machine Learning Models**:
   - Linear Regression: Simple and efficient for deployment.
   - GRU and Bidirectional LSTM: Capture temporal patterns for workload prediction.
   - Implemented in Python.
4. **SimPy Datacenter Simulation**:
   - Models real-time resource allocation in a datacenter environment.

---

## **Technologies Used**
| **Category**         | **Tools/Technologies**                            |
|----------------------|--------------------------------------------------|
| Cloud Services       | GCP (Cloud Run, Firebase Hosting, Cloud Storage) |
| Programming Languages| Python, JavaScript                               |
| ML Libraries         | TensorFlow, Keras, scikit-learn                  |
| Frontend Framework   | React, Chart.js                                  |
| Backend Framework    | Node.js, Express                                 |
| Simulation           | SimPy                                            |

---

## **Getting Started**

### **Prerequisites**
- [Node.js](https://nodejs.org/) (v16 or later)
- [Python](https://www.python.org/) (v3.8 or later)
- [Google Cloud SDK](https://cloud.google.com/sdk)
- Firebase CLI

### **Installation**

#### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/cloud-resource-optimization.git
cd cloud-resource-optimization
```

#### **2. Backend Setup**
```bash
cd cloud-resource-backend
npm install
```

#### **3. Frontend Setup**
```bash
cd ../cloud-resource-frontend
npm install
```

#### **4. ML Model Setup**
- Navigate to the `ml-model` directory.
- Install Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## **Usage**

### **Local Testing**
#### **Run the Backend**
```bash
cd cloud-resource-backend
node server.js
```

#### **Run the Frontend**
```bash
cd ../cloud-resource-frontend
npm start
```

#### **Run ML Predictions**
```bash
cd ../ml-model
python prediction_model.py
```

### **Deployment**

#### **Deploy Backend to Cloud Run**
1. Build and push Docker image:
   ```bash
   docker build -t gcr.io/<YOUR_PROJECT_ID>/cloud-backend .
   docker push gcr.io/<YOUR_PROJECT_ID>/cloud-backend
   ```
2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy cloud-backend --image gcr.io/<YOUR_PROJECT_ID>/cloud-backend --platform managed --region <YOUR_REGION> --allow-unauthenticated
   ```

#### **Deploy Frontend to Firebase Hosting**
1. Build the React app:
   ```bash
   npm run build
   ```
2. Deploy:
   ```bash
   firebase deploy
   ```

---

## **Project Highlights**
1. Achieved a **37.04% improvement** in time and energy metrics compared to traditional resource allocation methods.
2. Deployed a scalable and accessible system on **GCP**, demonstrating real-world applicability.
3. Integrated predictive ML models for proactive workload management, reducing scheduling overhead.
4. Developed a user-friendly **React dashboard** for visualizing CPU workloads and VM allocations.

---

## **Future Work**
- Enhance prediction accuracy by incorporating additional features (e.g., memory usage, I/O).
- Scale the SimPy simulation to model larger datacenters.
- Integrate energy optimization metrics for green cloud computing.
- Extend the system for multi-cloud compatibility (AWS, Azure).

---

## **Acknowledgments**
This project was developed under the guidance of **Prof. Fengguang Song** with support from **TAs Baixi Sun and Boyuan Zhang**. Their insights and feedback were invaluable in shaping the direction of this work.
