const Options = (props) => {
    let list = [];      //Maybe change this to arraylist if needed
    var data = require('./results.json'); 
    const handleClick = (e) => {
        console.log("Button clicked");
        window.data = data.trained.squeezenet1_1;
        console.log(window.data)
        //Here we can use the json object to create a list of all models and then 
        // using JSX we down below we can do a loop that will iterate through and display each one in a drop down box.
        //TODO: get models from a json file and add them to list. 
    }
    //console.log(props)
    
    return (
      <div>
        <div className="dropdown">
        <button className="dropbtn" onClick={handleClick}>CLICK MEH</button> 
          <button className="dropbtn">Select Model</button>
          <div className="dropdown-content">
            <a href="#">vgg19</a>
            <a href="#">squeezenet1_1</a>
            <a href="#">squeezenet1_0</a>
            <a href="#">resnet18</a>
            <a href="#">densenet161</a>
            <a href="#">densenet121</a>
            <a href="#">vgg13</a>
          </div>
        </div>
      </div>
    )
    //TODO: make that button a drop-down box displaying all options in List
}
export default Options;


/*
<div className="dropdown">
          <button className="dropbtn">Select Model</button>
          <div className="dropdown-content">
            <a href="#">vgg19</a>
            <a href="#">squeezenet1_1</a>
            <a href="#">squeezenet1_0</a>
            <a href="#">resnet18</a>
            <a href="#">densenet161</a>
            <a href="#">densenet121</a>
            <a href="#">vgg13</a>
          </div>
        </div>
*/