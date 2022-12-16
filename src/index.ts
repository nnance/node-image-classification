import { MnistDataset } from "./data";

const data = new MnistDataset();
data.loadData().then(() => {
  console.log(data.getTrainData());
  console.log(data.getTestData());
});
