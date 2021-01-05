var currentModel = "LR";

const select = (model) => {
  // Remove selected from all classes
  let models = ["LR", "DNN", "KNN", "CNN"];

  for (var i = 0; i < models.length; i++) {
    let identifier = models[i] + "-button";
    let buttonDOM = document.getElementById(identifier);
    buttonDOM.classList.remove("selected");
  }

  let identifier = model + "-button";
  let buttonDOM = document.getElementById(identifier);
  buttonDOM.classList.add("selected");

  currentModel = model;
  console.log(currentModel);
  updatePredictions();
};
