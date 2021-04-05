import { Bar } from "react-chartjs-2";
import {Component} from "react";



var data = require('./results.json'); 
var trained_models = [];
var StateOfTheArtModels = [];


for(var i in data.trained){
    trained_models.push(i);             //add names of trained models to list to display model names (doing this so its dynamic)
}
for(var i in data.soa){
    StateOfTheArtModels.push(i);             //add names of trained models to list to display model names (doing this so its dynamic)
}
var results = [data.trained.densenet121.cifar10.accuracy[0],data.trained.densenet121.cifar10.accuracy[1],data.trained.densenet121.cifar100.accuracy[0], data.trained.densenet121.cifar100.accuracy[1]]; //first 2 elements for cifar10, 3rd and 4th for cifar 100
var t_test_vals = [data.trained.densenet121.cifar10.t_test[0],data.trained.densenet121.cifar10.t_test[1],data.trained.densenet121.cifar100.t_test[0], data.trained.densenet121.cifar100.t_test[1]]; //first 2 elements for cifar10, 3rd and 4th for cifar 100
//console.log(t_test_vals);
var model_name = "densenet121";
export default class GraphBox extends Component {
    handleTrained = (e) => {
        //console.log(e)
        model_name = e.target.innerText;
        window.data = data.trained[model_name];
        results = [window.data.cifar10.accuracy[0],window.data.cifar10.accuracy[1],window.data.cifar100.accuracy[0], window.data.cifar100.accuracy[1]]; //first 2 elements for cifar10, 3rd and 4th for cifar 100
        t_test_vals = [window.data.cifar10.t_test[0],window.data.cifar10.t_test[1],window.data.cifar100.t_test[0], window.data.cifar100.t_test[1]];
        //console.log(results)
        this.setState(results)
    }
    handleSOA = (e) => {
        //console.log(e)
        model_name = e.target.innerText;
        window.data = data.soa[model_name];
        results = [window.data.cifar10.accuracy[0],window.data.cifar10.accuracy[1],window.data.cifar100.accuracy[0], window.data.cifar100.accuracy[1]]; //first 2 elements for cifar10, 3rd and 4th for cifar 100
        t_test_vals = [window.data.cifar10.t_test[0],window.data.cifar10.t_test[1],window.data.cifar100.t_test[0], window.data.cifar100.t_test[1]];
        //console.log(results)
        this.setState(results)
    }
    render() { return (
        <div > 
            <div className="dropdown"> 
                <button className="dropbtn">Select A Trained Model</button>
                <div className="dropdown-content">
                    {trained_models.map((model, i) => <button onClick={this.handleTrained}>{model}</button>)}       
                </div>
            </div>
            <div className="dropdown"> 
                <button className="dropbtn">Select A State of the Art Model</button>
                <div className="dropdown-content">
                    {StateOfTheArtModels.map((model, i) => <button onClick={this.handleSOA}>{model}</button>)}       
                </div>
            </div>
            <div > 
                <div className='graph-container'>
                    <div className="graph">
                        Accuracy Graph
                        <Bar
                            data={{
                                labels: ['CIFAR-10: Accuracy 1', 'CIFAR-10: Accuracy 5', 'CIFAR-100: Accuracy 1', 'CIFAR-100: Accuracy 5'],
                                datasets: [{
                                        label: window.data.name,
                                        data: results,   
                                        backgroundColor: [
                                            'rgba(255, 99, 132, 0.2)',
                                            'rgba(255, 99, 132, 0.2)',
                                            'rgba(54, 162, 235, 0.2)',
                                            'rgba(54, 162, 235, 0.2)'
                                        ],
                                        borderColor: [
                                            'rgba(255, 99, 132, 1)',
                                            'rgba(255, 99, 132, 1)',
                                            'rgba(54, 162, 235, 1)',
                                            'rgba(54, 162, 235, 1)'
                                        ],
                                        borderWidth: 1
                                }]
                            }}
                            height={200}
                            width={300}
                            options={{ maintainAspectRatio: false,
                            legend:
                                {
                                    display: false
                                },
                                scales: {
                                    yAxes: [{
                                        ticks: {
                                            beginAtZero: true, 
                                            min: 0,
                                            max: 100,
                                            scaleOverride: true,
                                            scaleSteps: 10,
                                            scaleStepWidth: 10,
                                        }
                                    }]
                                }
                            }}            //scalability
                        />
                        </div>
                        <div className="graph">
                        T-Test Graph
                        <Bar
                            data={{
                                labels: ['CIFAR-10: T-Test Val 1', 'CIFAR-10: T-Test Val 5', 'CIFAR-100: T-Test Val 1', 'CIFAR-100: T-Test Val 5'],
                                datasets: [{
                                        label: window.data.name,
                                        data: t_test_vals,   
                                        backgroundColor: [
                                            'rgba(255, 99, 132, 0.2)',
                                            'rgba(255, 99, 132, 0.2)',
                                            'rgba(54, 162, 235, 0.2)',
                                            'rgba(54, 162, 235, 0.2)'
                                        ],
                                        borderColor: [
                                            'rgba(255, 99, 132, 1)',
                                            'rgba(255, 99, 132, 1)',
                                            'rgba(54, 162, 235, 1)',
                                            'rgba(54, 162, 235, 1)'
                                        ],
                                        borderWidth: 1
                                }]
                            }}
                            height={200}
                            width={300}
                            options={{ maintainAspectRatio: false,
                            legend:
                                {
                                    display: false
                                },
                                scales: {
                                    yAxes: [{
                                        ticks: {
                                            beginAtZero: true
                                        }
                                    }]
                                }
                            }}            //scalability
                        />
                        </div>
                    </div>
                <div className="details-container">
                    <div className="detail">
                        <p><i>{ model_name }: </i> { window.data.details }</p>
                    
                    </div>
                    <div className="detail">
                        <p><i>Analysis: </i> { window.data.reason }</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
}

