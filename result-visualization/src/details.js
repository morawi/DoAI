const Details = (props) => {
    /*
        This is the most basic template, we should have scrollable lists (i.e. u can scroll in the box and the box will scroll not the whole 
        webpage). I think this is better for the user interface as it makes it much cleaner.
        We will also need to pass the object from the json file as a prop in order to provide details fo the model as T-Test data
        The variable test is just used here to show you how to access json object in JSX using react (even tho im pretty sure its done
        this exact way in regular JSX aswell.)

        TODO: Implement the data of the JSON object (model) to be displayed
        TODO: Implement Scrollable list 
    */
   var test = {
        model: "Model 1",
        t_test: "T-Test data will be displayed here",
        reason: "The data displayed is because ..."
   }
    return (
       <div className="details-container">
           <div className="detail">
               <p><i>Model: </i> { test.model}</p>
               
           </div>
           <div className="detail">
               <p><i>T-Test: </i>{ test.t_test }</p>
           </div>
           <div className="detail">
               <p><i>Reason: </i> { test.reason }</p>
           </div>
       </div>
    );
}
export default Details;