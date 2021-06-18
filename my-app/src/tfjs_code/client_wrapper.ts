import { ClothandMaskPath } from "./base_tfjs";

export class Client_Wrapper {
  tryonMap: Map<string, string[]>;
  personKey: string | null;
  constructor() {
    this.tryonMap = new Map<string, string[]>();
    this.personKey = null;
  }

  async setPersonKey() {
    let personKeyResults = await this._getPersonKeyServer();
    if (personKeyResults === null) {
      this.personKey = personKeyResults;
    } else {
      this.personKey = personKeyResults[0];
    }
  }

  async _getPersonKeyServer(): Promise<string[] | null> {
    return ["a", "b", "c"];
  }

  async _runTryonServer(
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
    this.personKey = person_key;
    let cloth_path = clothsAndMasksPath[0][0];
    let cloth_mask_path = clothsAndMasksPath[0][1];
    let cloth_key = cloth_path + ":" + cloth_mask_path;
    let tryon_key = cloth_key + ":" + person_key;
    if (this.tryonMap.has(tryon_key)) {
      return this.tryonMap.get(tryon_key)!;
    } else {
      let tryonResult = await this._runTryonServer(
        clothsAndMasksPath,
        person_key,
        person_data_url
      );
      this.tryonMap.set(tryon_key, tryonResult);
      return tryonResult;
    }
  }
}
