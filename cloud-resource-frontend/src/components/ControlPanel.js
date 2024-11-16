import React, { useState } from 'react';

function ControlPanel() {
    const [numVMs, setNumVMs] = useState(10);
    const [numServers, setNumServers] = useState(5);

    const handleSubmit = () => {
        alert(`Simulation started with ${numVMs} VMs and ${numServers} servers.`);
    };

    return (
        <div>
            <h2>Control Panel</h2>
            <label>
                Number of VMs:
                <input
                    type="number"
                    value={numVMs}
                    onChange={(e) => setNumVMs(e.target.value)}
                />
            </label>
            <br />
            <label>
                Number of Servers:
                <input
                    type="number"
                    value={numServers}
                    onChange={(e) => setNumServers(e.target.value)}
                />
            </label>
            <br />
            <button onClick={handleSubmit}>Run Simulation</button>
        </div>
    );
}

export default ControlPanel;
