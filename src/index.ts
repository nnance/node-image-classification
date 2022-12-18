import * as argparse from "argparse";
import { trainSave } from "./execution";
import { model } from "./model";
import { data } from "./data";

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

trainSave(model, data, args.epochs, args.batch_size, args.model_save_path);
