const Options = () => {
    let list = [];      //Maybe change this to arraylist if needed
    const handleClick = (e) => {
        console.log("Button clicked");
        //Here we can use the json object to create a list of all models and then 
        // using JSX we down below we can do a loop that will iterate through and display each one in a drop down box.
        //TODO: get models from a json file and add them to list. 
    }
    return (
    <div className ="options-container">
        <button onClick={handleClick}> Pick a model </button>
    </div>
    )
    //TODO: make that button a drop-down box displaying all options in List
}
export default Options;