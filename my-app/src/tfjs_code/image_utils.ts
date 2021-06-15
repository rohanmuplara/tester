export async function downloadImage(image_url: string) {
  let image = new Image();
  image.crossOrigin = "anonymous";
  let image_promise = onload2promise(image);
  image.src = image_url;
  await image_promise;
  return image;
}

export async function convert_file_to_img(file: any) {
  let image = new Image();
  let fr = new FileReader();
  fr.onload = function () {
    image.src = fr.result as string;
  };
  let image_promise = onload2promise(image);
  fr.readAsDataURL(file);
  await image_promise;
  return image_promise;
}

interface OnLoadAble {
  onload: any;
}
function onload2promise<T extends OnLoadAble>(obj: T): Promise<T> {
  return new Promise((resolve, reject) => {
    obj.onload = () => resolve(obj);
  });
}
