import './App.css';
import Header from './header.js'
import GraphBox from './graphs.js'

function App() {
  var data = require('./results.json');
  window.data = data.trained.densenet121;
  return (
    <div className="App">
      <Header/>     
      <GraphBox />
    </div>
  );
}

export default App;
