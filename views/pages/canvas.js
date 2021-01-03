const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1;


const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button")


let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0

const sess = new onnx.InferenceSession()
const loadingModelPromise = sess.loadModel('./my-model.onnx')

ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#212121";

ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2)

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col"
    element.children[0].children[0].style.height = "0"
  }
}


function drawLine(fromX, fromY, toX, toY) {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}


async function updatePredictions() {
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE)
  const inputTwo = new onnx.Tensor(new Float32Array(imgData.data), "float32");
  const input = new onnx.Tensor(new Float32Array(280 * 280 * 4), 'float32', [313600])
  // console.log(inputTwo)
  const outputMap = await sess.run([inputTwo]);

  const outputTensor = outputMap.values().next().value;
  const predictions = outputTensor.data
  const maxPrediction = Math.max(...predictions);

  for (let i = 0; i < predictions.length; i++) {
    const element = document.getElementById(`prediction-${i}`);

    element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    element.className =
      predictions[i] === maxPrediction ?
      "prediction-col top-prediction" :
      "prediction-col";
  }
}

function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false
  }

  const x = event.offsetX / CANVAS_SCALE
  const y = event.offsetY / CANVAS_SCALE

  lastX = x + 0.001
  lastY = y + 0.001
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE
  const y = event.offsetY / CANVAS_SCALE

  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }

  lastX = x;
  lastY = y;
}

function canvasTouchStart(event) {
  mousePos = getTouchPos(canvas, event);
  let touch = event.touches[0];
  let mouseEvent = new MouseEvent("mousedown", {
    clientX: touch.clientX,
    clientY: touch.clientY
  })
  canvas.dispatchEvent(mouseEvent)
}

function canvasTouchEnd(event) {
  var mouseEvent = new MouseEvent("mouseup", {});
  canvas.dispatchEvent(mouseEvent);
}

function canvasTouchMove(event) {
  var touch = event.touches[0];
  var mouseEvent = new MouseEvent("mousemove", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}

function bodyMouseUp() {
  isMouseDown = false;
}


function bodyMouseOut(event) {
  // We won't be able to detect a MouseUp event if the mouse has move
  // Outside the window, so when the mouse leaves the window,
  // we set ismousedown to false
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false
  }
}
// Get the position of a touch relative to the canvas
function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: touchEvent.touches[0].clientX,
    y: touchEvent.touches[0].clientY - rect.top
  }
};





loadingModelPromise.then(() => {
  canvas.addEventListener("mousedown", canvasMouseDown)
  canvas.addEventListener("mousemove", canvasMouseMove)
  canvas.addEventListener("touchstart", canvasTouchStart, false);
  canvas.addEventListener("touchend", canvasTouchEnd, false);
  canvas.addEventListener("touchmove", function(e) {
    var touch = e.touches[0];
    var mouseEvent = new MouseEvent("mousemove", {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  }, false);


  document.body.addEventListener("mouseup", bodyMouseUp)
  document.body.addEventListener("mouseout", bodyMouseOut)
  clearButton.addEventListener("mousedown", clearCanvas)


  document.body.addEventListener("touchstart", function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }
  }, false);
  document.body.addEventListener("touchend", function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }
  }, false);
  document.body.addEventListener("touchmove", function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }
  }, false);


  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
  ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2)
})
