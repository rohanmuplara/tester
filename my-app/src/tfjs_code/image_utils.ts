export async function downloadImages(
  image_urls: string[]
): Promise<HTMLImageElement[]> {
  return await Promise.all(
    image_urls.map(async (image_url) => {
      let image = new Image();
      image.crossOrigin = "anonymous";
      let image_promise = onload2promise(image);
      image.src = image_url;
      await image_promise;
      return image;
    })
  );
}

export async function convert_files_to_img_data(
  files: File[]
): Promise<string[]> {
  return await Promise.all(
    files.map(async (file) => {
      let fileReader = new FileReader();
      let fileReaderPromise = onload2promise(fileReader);
      fileReader.readAsDataURL(file);
      await fileReaderPromise;
      return fileReader.result as string;
    })
  );
}

interface OnLoadAble {
  onload: any;
}
function onload2promise<T extends OnLoadAble>(obj: T): Promise<T> {
  return new Promise((resolve, reject) => {
    obj.onload = () => resolve(obj);
  });
}
