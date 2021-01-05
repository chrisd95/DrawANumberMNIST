const express = require("express");
const path = require("path");
const PORT = process.env.PORT || 5000;

express()
  .use(express.static(path.join(__dirname, "public")))
  .set("views", path.join(__dirname, "views"))
  .set("view engine", "ejs")
  .get("/", (req, res) => res.render("pages/index"))
  .get("/my-model-logistic-regression.onnx", (req, res) =>
    res.sendfile("views/pages/onnx_model_logistic_regression.onnx")
  )
  .get("/my-model-deep-neural-network.onnx", (req, res) =>
    res.sendfile("views/pages/onnx_model_deep_neural_network.onnx")
  )
  .get("/canvas.js", (req, res) => res.sendfile("views/pages/canvas.js"))
  .get("/dynamic-style.js", (req, res) =>
    res.sendfile("views/pages/dynamic-style.js")
  )

  .listen(PORT, () => console.log(`Listening on ${PORT}`));
