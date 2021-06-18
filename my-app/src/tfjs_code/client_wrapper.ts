import { ClothandMaskPath } from "./base_tfjs";

export class Client_Wrapper {
  tryonMap: Map<string, string[]>;
  constructor() {
    this.tryonMap = new Map<string, string[]>();
  }

  getExistingPersonKey(): string | null {
    return null;
  }

  runTryonServer(
    clothsAndMasksPath: ClothandMaskPath[],
    person_key: string,
    person_data_url?: string
  ) {
    return [""];
  }

  async runTryonClient(
    clothsAndMasksPath: ClothandMaskPath[],
    person_key: string,
    person_data_url?: string
  ): Promise<string[]> {
    let cloth_path = clothsAndMasksPath[0][0];
    let cloth_mask_path = clothsAndMasksPath[0][1];
    let cloth_key = cloth_path + ":" + cloth_mask_path;
    let tryon_key = cloth_path + ":" + person_key;
    if (this.tryonMap.has(person_key)) {
      return this.tryonMap.get(person_key)!;
    } else {
      let tryonResult = this.runTryonServer(
        clothsAndMasksPath,
        person_key,
        person_data_url
      );
      this.tryonMap.set(tryon_key, tryonResult);
      return tryonResult;
    }
  }
}
