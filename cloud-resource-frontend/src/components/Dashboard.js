import React from 'react';
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, LineController } from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register the required components for Chart.js
ChartJS.register(LineElement, LineController, CategoryScale, LinearScale, PointElement);

function Dashboard() {
    const data = {
        labels: ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5'],
        datasets: [
            {
                label: 'Predicted CPU Usage (%)',
                data: [20, 30, 50, 70, 60],
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
            <Line data={data} options={options} />
        </div>
    );
}

export default Dashboard;



//import React, { useEffect, useRef } from 'react';
//import { Chart } from 'chart.js';
//
//function Dashboard() {
//    const chartRef = useRef(null);
//
//    useEffect(() => {
//        const ctx = chartRef.current.getContext('2d');
//        const chartInstance = new Chart(ctx, {
//            type: 'line',
//            data: {
//                labels: ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5'],
//                datasets: [
//                    {
//                        label: 'Predicted CPU Usage (%)',
//                        data: [20, 30, 50, 70, 60],
//                        borderColor: 'blue',
//                        fill: false,
//                    },
//                ],
//            },
//        });
//
//        return () => {
//            chartInstance.destroy(); // Cleanup previous chart instance
//        };
//    }, []);
//
//    return <canvas ref={chartRef}></canvas>;
//}
//
//export default Dashboard;
