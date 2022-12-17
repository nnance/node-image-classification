import * as argparse from "argparse";

import { data } from "./data";
import { model } from "./model";

async function run(epochs: number, batchSize: number, modelSavePath: string) {
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

const parser = new argparse.ArgumentParser({
  description: "TensorFlow.js-Node MNIST Example.",
  add_help: true,
});
parser.add_argument("--epochs", {
  type: "int",
  default: 20,
  help: "Number of epochs to train the model for.",
});
parser.add_argument("--batch_size", {
  type: "int",
  default: 128,
  help: "Batch size to be used during model training.",
});
parser.add_argument("--model_save_path", {
  type: "string",
  help: "Path to which the model will be saved after training.",
});
const args = parser.parse_args();

run(args.epochs, args.batch_size, args.model_save_path);
