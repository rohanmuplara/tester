import * as tf from "@tensorflow/tfjs-core";
import { Storage_Map } from "./storage_map";
import { convertDataUrlsToTensor, converTensorToDataUrls } from "./core";
import { NamedTensor4DMap } from "./base_tfjs";

export class Tensor_Storage_Map extends Storage_Map {
  async setNameTensorMap(key: string, namedTensorMap: NamedTensor4DMap) {
    let nameArrayMap = await Promise.all(
      Object.entries(namedTensorMap).map(async ([key, tensor]) => {
        let dataUrls = await converTensorToDataUrls(tensor);
        let lastDimensionShape = tensor.shape[3];
        return [
          key,
          { lastDimensionShape: lastDimensionShape, dataUrls: dataUrls },
        ];
      })
    ).then(Object.fromEntries);
    await this.setItem(key, nameArrayMap);
  }
  async getNamedTensorMap(key: string): Promise<NamedTensor4DMap | null> {
    let nameValueMap = await this.getItem(key);
    if (nameValueMap) {
      let nameTensorEntries = await Promise.all(
        Object.entries(nameValueMap).map(
          async ([key, serializedDict]: [string, any]) => {
            let lastDimensionShape = serializedDict["lastDimensionShape"];
            let dataUrls = serializedDict["dataUrls"];
            let tensor = await convertDataUrlsToTensor(dataUrls);
            let newTensor;
            if (lastDimensionShape === 1) {
              let tensorList = tf.split(tensor, 3, 3);
              newTensor = tensorList[0];
              tf.dispose(tensorList[1]);
              tf.dispose(tensorList[2]);

              tf.dispose(tensor);
            } else {
              newTensor = tensor;
            }
            return [key, newTensor];
          }
        )
      );
      return Object.fromEntries(nameTensorEntries) as NamedTensor4DMap;
    } else {
      return null;
    }
  }
}
