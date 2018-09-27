const tf = require("@tensorflow/tfjs");

const irish_testing = require("./irish_testing.json");
const irish = require("./irish.json");
const training = tf.tensor2d(
  irish.map(i => [i.sepal_length, i.sepal_width, i.petal_length, i.petal_width])
);
const output = tf.tensor2d(
  irish.map(i => [
    i.species === "setosa" ? 1 : 0,
    i.species === "virginica" ? 1 : 0,
    i.species === "versicolor" ? 1 : 0
  ])
);
const testing = tf.tensor2d(
  irish_testing.map(i => [
    i.sepal_length,
    i.sepal_width,
    i.petal_length,
    i.petal_width
  ])
);

const model = tf.sequential();
model.add(
  tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5
  })
);
model.add(
  tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3
  })
);
model.add(
  tf.layers.dense({
    activation: "sigmoid",
    units: 3
  })
);
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(0.06)
});

model.fit(training, output, { epochs: 100 }).then(history => {
  console.log("Done", history);

  model.predict(testing).print();
});
