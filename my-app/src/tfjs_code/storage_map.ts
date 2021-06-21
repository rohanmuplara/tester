export class Storage_Map {
  existing_keys: string[];
  namespace: string;
  maxItems: number;
  constructor(namespace: string, maxItems: number) {
    this.namespace = namespace;
    let metaTablePath = namespace + ":metadata";
    let metaTableData = localStorage.getItem(metaTablePath);
    if (metaTableData === null) {
      this.existing_keys = [];
    } else {
      this.existing_keys = JSON.parse(metaTableData);
    }
    this.maxItems = maxItems;
  }
  setItem(key: string, value: any): void {
    let namespacedKey = this.namespace + ":" + key;
    let jsonValue = JSON.stringify(value);
    localStorage.setItem(namespacedKey, jsonValue);
    if (this.existing_keys.length === this.maxItems) {
      let removal_key = this.existing_keys[-1];
      let removal_namespaced_key = this.namespace + ":" + removal_key;
      localStorage.removeItem(removal_namespaced_key);
    }
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
