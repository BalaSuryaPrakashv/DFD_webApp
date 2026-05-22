const fileInput = document.querySelector('input[type="file"]');
const previewContainer = document.getElementById('preview-container');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  const reader = new FileReader();
  
  reader.addEventListener('load', () => {
    const previewElement = document.createElement(getPreviewType(file));
    previewElement.src = reader.result;
    previewContainer.appendChild(previewElement);
  });
  
  reader.readAsDataURL(file);
});

function getPreviewType(file) {
  const type = file.type.split('/')[0];
  if (type === 'image') {
    return 'img';
  } else if (type === 'video') {
    return 'video';
  } else {
    throw new Error(`Invalid file type: ${type}`);
  }
}
