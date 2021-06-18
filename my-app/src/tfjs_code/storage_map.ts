export class Storage_Map {
  existing_keys: string[];
  namespace: string;
  constructor(namespace: string) {
    this.namespace = namespace;
    let metaTablePath = namespace + ":metadata";
    let metaTableData = localStorage.getItem(metaTablePath);
    if (metaTableData === null) {
      this.existing_keys = [];
    } else {
      this.existing_keys = JSON.parse(metaTableData);
    }
  }
  setItem(key: string, value: any): void {
    let namespacedKey = this.namespace + ":" + key;
    let jsonValue = JSON.stringify(value);
    localStorage.setItem(namespacedKey, jsonValue);
    this.existing_keys.unshift(key);
    let metaTablePath = this.namespace + ":metadata";
    localStorage.setItem(metaTablePath, JSON.stringify(this.existing_keys));
  }
  getItem(key: string): any {
    let namespaced_key = this.namespace + ":" + key;
    let json_value = localStorage.getItem(namespaced_key);
    if (json_value) {
      return JSON.parse(json_value!);
    } else {
      return null;
    }
  }

  getExistingKeys(): string[] {
    return this.existing_keys;
  }
}
