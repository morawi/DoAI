import { Bar } from "react-chartjs-2"

const GraphBox = () => {
    var data = require('./results.json');
    var results = [data.vgg19.cifar10.results[0],data.vgg19.cifar10.results[1],data.vgg19.cifar100.results[0], data.vgg19.cifar100.results[1]]; //first 2 elements for cifar10, 3rd and 4th for cifar 100
    return (
        <div className="graph-container">
            <div className="graph">
                <Bar
                    data={{
                        labels: ['CIFAR-10: Accuracy 1', 'CIFAR-10: Accuracy 5', 'CIFAR-100: Accuracy 1', 'CIFAR-100: Accuracy 5'],
                        datasets: [{
                                label: 'vgg19',
                                data: [9.6848, 48.52, 1.04, 4.9],   
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
                    scales: {
                        yAxes: [{
                            ticks: {
                                beginAtZero: true, 
                            }
                        }]
                    }
                 }}            //scalability
                />
            </div>
            <div className="graph">
                <h3> T-Test Statistics</h3>
                <p><b>Cifar 10:</b></p>
                <p> T-Test value is {data.vgg19.cifar10.t_test}</p>
                <p> P-value is {data.vgg19.cifar10.p_val}</p>
                <p><b>Cifar 100:</b></p>
                <p> T-Test value is {data.vgg19.cifar100.t_test}</p>
                <p> P-value is {data.vgg19.cifar100.p_val}</p>
            </div>
        </div>
    )
}
export default GraphBox;