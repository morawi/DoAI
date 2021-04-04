import './App.css';
import Header from './header.js'
import GraphBox from './graphs.js'
import Options from './options.js'
import Details from "./details.js"



function App() {
  return (
    <div className="App">
      <Header/>
      <Options/>      
      <GraphBox />
      
      <Details />
    </div>
  );
}

export default App;
