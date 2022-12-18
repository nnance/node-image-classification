import * as tf from "@tensorflow/tfjs-node";
import { MnistDataset } from "./data";

export async function trainSave(
  model: tf.Sequential,
  data: MnistDataset,
  epochs: number,
  batchSize: number,
  modelSavePath: string
) {
  await data.loadData();

  const { images: trainImages, labels: trainLabels } = data.getTrainData();
  model.summary();

  const validationSplit = 0.15;
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
  });

  const { images: testImages, labels: testLabels } = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  if (!Array.isArray(evalOutput)) return;

  console.log(
    `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`
  );

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

function doPrediction(
  model: tf.Sequential,
  data: MnistDataset,
  testDataSize = 500
) {
  const { images, labels } = data.getTestData();
  const labeled = labels.argMax(-1);
  const predResults = model.predict(images);
  const preds = Array.isArray(predResults)
    ? predResults[0].argMax(-1)
    : predResults.argMax(-1);

  return [preds, labeled];
}

export async function showAccuracy(model: tf.Sequential, data: MnistDataset) {
  const [preds, labels] = doPrediction(model, data);
  console.log(`Loss = ${preds.dataSync()[0].toFixed(3)}`);

  labels.dispose();
}

export async function showConfusion(model: tf.Sequential, data: MnistDataset) {
  const [preds, labels] = doPrediction(model, data);
  console.log(`Accuracy = ${preds.dataSync()[1].toFixed(3)}`);
  labels.dispose();
}
