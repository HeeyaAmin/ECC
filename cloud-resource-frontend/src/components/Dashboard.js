import React, { useEffect, useState } from 'react';
import {
    Chart as ChartJS,
    LineElement,
    CategoryScale,
    LinearScale,
    PointElement,
    LineController,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

// Register the required components for Chart.js
ChartJS.register(LineElement, LineController, CategoryScale, LinearScale, PointElement);

function Dashboard() {
    const [cpuData, setCpuData] = useState([]);

    // Fetch data from the backend API
    useEffect(() => {
        axios
            .get('http://localhost:5000/api/predict') // Replace with your API endpoint
            .then((response) => {
                setCpuData(response.data); // Assume the API returns an array of CPU usage values
            })
            .catch((error) => {
                console.error('Error fetching CPU data:', error);
            });
    }, []);

    // Prepare chart data
    const chartData = {
        labels: ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5'], // Replace with dynamic labels if available
        datasets: [
            {
                label: 'Predicted CPU Usage (%)',
                data: cpuData.length ? cpuData : [0, 0, 0, 0, 0], // Use fetched data or fallback to placeholder
                borderColor: 'blue',
                borderWidth: 2,
                fill: false,
            },
        ],
    };

    const options = {
        responsive: true,
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Time (Hours)',
                },
            },
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'CPU Usage (%)',
                },
            },
        },
    };

    return (
        <div>
            <h2>Dashboard</h2>
            <Line data={chartData} options={options} />
        </div>
    );
}

export default Dashboard;






//import React from 'react';
//import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, LineController } from 'chart.js';
//import { Line } from 'react-chartjs-2';
//
//// Register the required components for Chart.js
//ChartJS.register(LineElement, LineController, CategoryScale, LinearScale, PointElement);
//
//function Dashboard() {
//    const data = {
//        labels: ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5'],
//        datasets: [
//            {
//                label: 'Predicted CPU Usage (%)',
//                data: [20, 30, 50, 70, 60],
//                borderColor: 'blue',
//                borderWidth: 2,
//                fill: false,
//            },
//        ],
//    };
//
//    const options = {
//        responsive: true,
//        scales: {
//            x: {
//                title: {
//                    display: true,
//                    text: 'Time (Hours)',
//                },
//            },
//            y: {
//                beginAtZero: true,
//                title: {
//                    display: true,
//                    text: 'CPU Usage (%)',
//                },
//            },
//        },
//    };
//
//    return (
//        <div>
//            <h2>Dashboard</h2>
//            <Line data={data} options={options} />
//        </div>
//    );
//}
//
//export default Dashboard;
//
//
//
////import React, { useEffect, useRef } from 'react';
////import { Chart } from 'chart.js';
////
////function Dashboard() {
////    const chartRef = useRef(null);
////
////    useEffect(() => {
////        const ctx = chartRef.current.getContext('2d');
////        const chartInstance = new Chart(ctx, {
////            type: 'line',
////            data: {
////                labels: ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5'],
////                datasets: [
////                    {
////                        label: 'Predicted CPU Usage (%)',
////                        data: [20, 30, 50, 70, 60],
////                        borderColor: 'blue',
////                        fill: false,
////                    },
////                ],
////            },
////        });
////
////        return () => {
////            chartInstance.destroy(); // Cleanup previous chart instance
////        };
////    }, []);
////
////    return <canvas ref={chartRef}></canvas>;
////}
////
////export default Dashboard;
