import { Bar, Line } from "react-chartjs-2";
import {Component} from "react";



var data = require('./results.json'); 
var trained_models = [];
var StateOfTheArtModels = [];
var labels = []


for(var i in data.trained){
    trained_models.push(i);             //add names of trained models to list to display model names (doing this so its dynamic)
}
for(var i in data.soa){
    StateOfTheArtModels.push(i);             //add names of trained models to list to display model names (doing this so its dynamic)
}
var i=0;
while (i<100)
  {
  i++;
  labels.push(i);
  }
console.log(labels)
var cifar10_acc1 = data.trained.densenet121.cifar10.acc_1; 
var cifar10_acc5 = data.trained.densenet121.cifar10.acc_5;
var cifar100_acc1 = data.trained.densenet121.cifar100.acc_1; 
var cifar100_acc5 = data.trained.densenet121.cifar100.acc_5;  
var t_test_vals = [data.trained.densenet121.cifar10.t_test[1],data.trained.densenet121.cifar10.t_test[0],data.trained.densenet121.cifar100.t_test[1], data.trained.densenet121.cifar100.t_test[0]]; //first 2 elements for cifar10, 3rd and 4th for cifar 100
//console.log(t_test_vals);
var model_name = "densenet121";
export default class GraphBox extends Component {
    handleTrained = (e) => {
        //console.log(e)
        model_name = e.target.innerText;
        window.data = data.trained[model_name];
        //console.log(window.data.cifar10)
        cifar10_acc1 = window.data.cifar10.acc_1; 
        cifar10_acc5 = window.data.cifar10.acc_5; 
        cifar100_acc1 = window.data.cifar100.acc_1; 
        cifar100_acc5 = window.data.cifar100.acc_5;
        t_test_vals = [window.data.cifar10.t_test[0],window.data.cifar10.t_test[1],window.data.cifar100.t_test[0], window.data.cifar100.t_test[1]];
        console.log(window.data)
        this.setState(cifar10_acc1)
    }
    handleSOA = (e) => {
        //console.log(e)
        model_name = e.target.innerText;
        window.data = data.soa[model_name];
        cifar10_acc1 = window.data.cifar10.acc_1; 
        cifar10_acc5 = window.data.cifar10.acc_5; 
        t_test_vals = [window.data.cifar10.t_test[0],window.data.cifar10.t_test[1],window.data.cifar100.t_test[0], window.data.cifar100.t_test[1]];
        //console.log(results)
        this.setState(cifar10_acc1)
    }
    render() { return (
        <div > 
            <div className="dropdown"> 
                <button className="dropbtn">Select A Pre-Trained Model</button>
                <div className="dropdown-content">
                    {trained_models.map((model, i) => <button onClick={this.handleTrained}>{model}</button>)}       
                </div>
            </div>
            <div className="dropdown"> 
                <button className="dropbtn">Select A Pre-Trained Model</button>
                <div className="dropdown-content">
                    {StateOfTheArtModels.map((model, i) => <button onClick={this.handleSOA}>{model}</button>)}       
                </div>
            </div>
            <h3> { model_name }</h3>
            <div > 
                <div className='graph-container'>
                    <div className="graph">
                        <Line
                            data={{
                                labels: labels,
                                datasets: [{
                                        label: model_name + " Cifar10 Acc1",
                                        data: cifar10_acc1,
                                        borderColor: [
                                            'rgba(255, 140, 140, 1)'
                                        ],
                                        fill: false,
                                        borderWidth: 3
                                    },{
                                        label: model_name + " Cifar10 Acc5",
                                        data: cifar10_acc5,
                                        borderColor: [
                                            'rgba(255, 0, 0, 1)'                                            
                                        ],
                                        fill: false,
                                        borderWidth: 3
                                    },{
                                        label: model_name + " Cifar100 Acc1",
                                        data: cifar100_acc1,
                                        borderColor: [
                                            'rgba(0, 255, 200, 1)'
                                        ],
                                        fill: false,
                                        borderWidth: 3
                                    },{
                                        label: model_name + " Cifar100 Acc5",
                                        data: cifar100_acc5,
                                        borderColor: [
                                            'rgba(0, 255, 255, 1)'                                            
                                        ],
                                        fill: false,
                                        borderWidth: 3
                                    }
                                ]   
                            }}
                            height={200}
                            width={300}
                            options={{ maintainAspectRatio: false,
                            legend:
                                {
                                    display: true
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
                        <Bar
                            data={{
                                labels: ['CIFAR-10: Accuracy 1', 'CIFAR-10: Accuracy 5', 'CIFAR-100: Accuracy 1', 'CIFAR-100: Accuracy 5'],
                                datasets: [{
                                        label: window.data.name,
                                        data: t_test_vals,   
                                        backgroundColor: [
                                            'rgba(255, 140, 140, 0.2)',
                                            'rgba(255, 0, 0, 0.2)',
                                            'rgba(54, 162, 235, 0.2)',
                                            'rgba(54, 162, 235, 0.2)'
                                        ],
                                        borderColor: [
                                            'rgba(255, 140, 140, 1)',
                                            'rgba(255, 0, 0, 1)',
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
                <p><i>Reason: </i> { window.data.reason }</p>
            </div>
        </div>
        </div>
    </div>
    )
}
}

