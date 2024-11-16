//import logo from './logo.svg';
//import './App.css';
//
//function App() {
//  return (
//    <div className="App">
//      <header className="App-header">
//        <img src={logo} className="App-logo" alt="logo" />
//        <p>
//          Edit <code>src/App.js</code> and save to reload.
//        </p>
//        <a
//          className="App-link"
//          href="https://reactjs.org"
//          target="_blank"
//          rel="noopener noreferrer"
//        >
//          Learn React
//        </a>
//      </header>
//    </div>
//  );
//}
//
//export default App;

import React from 'react';
import Dashboard from './components/Dashboard';
import ControlPanel from './components/ControlPanel';
import ComparisonResults from './components/ComparisonResults';


function App() {
//    console.log("print")
    return (
        <div>
            <h1>Cloud Resource Optimization</h1>
            <ControlPanel />
            <Dashboard />
            <ComparisonResults />
        </div>
    );
}

export default App;
