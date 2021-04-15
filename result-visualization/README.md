### download and install node.js
download node.js @ https://nodejs.org/en/ .\
Node.js is needed to use the npm command in terminal and to run react.js

### `cd DoAI`
cd into the local copy of this DoAI Git Repository

### `cd result-visualization`
Make the current repo this result-visualization folder. 

### `npm install`
Run the npm install command to install all the modules in the package.json file allowing you to run the web-page easily. .\
Install all modules in the package.json file in order to run the file .\
You may also need to install chokidar this is done by  simply running .\
`npm install chokidar`

### `npm start`
Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### USAGE
This front-end is entirely dynamic as we wanted to make it as easy as possible to add in a new models that have been trained or add in results for state of the art models. To add a new model into the front end simply update the results.JSON file in the src folder. The JSON is formatted in a way that will display the graphs and data appropriately so follow the format.\
    {
        "trained": {
            "model_name" : {
                "cifar10" : {
                    "acc_1" : [array of values for 100 epochs],
                    "acc_5" : [array of values for 100 epochs],
                    "t_test" : [array of 2 T-Test values]
                    },
                "cifar100" : {
                    "acc_1" : [array of values for 100 epochs],
                    "acc_5" : [array of values for 100 epochs],
                    "t_test" : [array of 2 T-Test values]
                    },
                "name" : "name of model as string",
                "details" : "details of the model as string i.e. what it is and how it works"
                "reason: : "reason for the result being the way it is as string"
                } 
            } 
        },
        "soa" : {
            "soa_example": {
                "cifar10": {
                    "accuracy": [ accuracy 1, accuracy 5],
                    "t_test": [array of 2 T-Test values]
                },
                "cifar100": {
                    "accuracy": [ accuracy 1, accuracy 5],
                    "t_test": [array of 2 T-Test values]
                },
                "name" : "name of model as string",
                "details": "details of the model as string i.e. what it is and how it works",
                "reason": "reason for the result being the way it is as string"
            }
        }
    }
If you are adding in a trained model insert it into the "trained" JSON object adn insert the data in as explained above. If you are inserting in a state of the art model insert it into the "soa" JSON object and follow the above format.