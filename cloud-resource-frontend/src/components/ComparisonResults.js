import React from 'react';

function ComparisonResults() {
    return (
        <div>
            <h2>Comparison Results</h2>
            <table border="1">
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>Energy Consumption</th>
                        <th>Processing Time</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>AI/ML</td>
                        <td>Low</td>
                        <td>Fast</td>
                    </tr>
                    <tr>
                        <td>Heuristic</td>
                        <td>High</td>
                        <td>Slow</td>
                    </tr>
                </tbody>
            </table>
        </div>
    );
}

export default ComparisonResults;
