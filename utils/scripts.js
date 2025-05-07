async () => {
    let images = [];
    let selectedImage = null;
    let offsetX = 0;
    let offsetY = 0;

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const fileInput = document.getElementById('fileInput');
    const chooseFileButton = document.getElementById('chooseFileButton');
    const processButton = document.getElementById('processButton');
    const bringToFrontButton = document.getElementById('bringToFrontButton');
    const sendToBackButton = document.getElementById('sendToBackButton');
    const exampleButton = document.getElementById('exampleButton');
    const scaleUpButton = document.getElementById('scaleUpButton');
    const scaleDownButton = document.getElementById('scaleDownButton');
    const flipHorizontalButton = document.getElementById('flipHorizontalButton');

    chooseFileButton.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
        handleFiles(event.target.files);
    });

    canvas.addEventListener('dragover', (event) => {
        event.preventDefault();
        canvas.classList.add('dragover');
    });

    canvas.addEventListener('dragleave', () => {
        canvas.classList.remove('dragover');
    });

    canvas.addEventListener('drop', (event) => {
        event.preventDefault();
        canvas.classList.remove('dragover');
        handleFiles(event.dataTransfer.files);
    });

    function handleFiles(files) {
        for (const file of files) {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = new Image();
                    img.src = e.target.result;
                    img.onload = () => {
                        let width = img.width;
                        let height = img.height;

                        if (width > 512 || height > 512) {
                            const aspectRatio = width / height;
                            if (width > height) {
                                width = 512;
                                height = Math.round(512 / aspectRatio);
                            } else {
                                height = 512;
                                width = Math.round(512 * aspectRatio);
                            }
                        }
                        images.push({
                            img: img,
                            base64: e.target.result,
                            x: 10,
                            y: 10,
                            width: width,
                            height: height,
                            flipX: false,
                        });
                        drawCanvas();
                    };
                };
                reader.readAsDataURL(file);
            }
        }
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        images.forEach(image => {
            ctx.save();
            if (image.flipX) {
                ctx.translate(image.x + image.width, image.y);
                ctx.scale(-1, 1);
                ctx.drawImage(image.img, 0, 0, image.width, image.height);
            } else {
                ctx.drawImage(image.img, image.x, image.y, image.width, image.height);
            }
            ctx.restore();
        });
    }

    canvas.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        for (let i = images.length - 1; i >= 0; i--) {
            const img = images[i];
            let hitX = x;
            if (img.flipX) {
                hitX = img.x + img.width - (x - img.x);
            }
            if (hitX >= img.x && hitX <= img.x + img.width && y >= img.y && y <= img.y + img.height) {
                images.splice(i, 1);
                drawCanvas();
                selectedImage = null;
                break;
            }
        }
    });

    canvas.addEventListener('mousedown', (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        selectedImage = null;
        for (let i = images.length - 1; i >= 0; i--) {
            const img = images[i];
            let hitX = x;
            if (img.flipX) {
                hitX = img.x + img.width - (x - img.x);
            }
            if (hitX >= img.x && hitX <= img.x + img.width && y >= img.y && y <= img.y + img.height) {
                selectedImage = img;
                offsetX = x - img.x;
                offsetY = y - img.y;
                break;
            }
        }
    });

    canvas.addEventListener('mousemove', (event) => {
        if (selectedImage && event.buttons === 1) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            selectedImage.x = x - offsetX;
            selectedImage.y = y - offsetY;
            drawCanvas();
        }
    });

    bringToFrontButton.addEventListener('click', () => {
        if (selectedImage) {
            const index = images.indexOf(selectedImage);
            if (index !== -1 && index < images.length - 1) {
                images.splice(index, 1);
                images.push(selectedImage);
                drawCanvas();
            }
        } else {
            alert('Please select an image first.');
        }
    });

    sendToBackButton.addEventListener('click', () => {
        if (selectedImage) {
            const index = images.indexOf(selectedImage);
            if (index !== -1 && index > 0) {
                images.splice(index, 1);
                images.unshift(selectedImage);
                drawCanvas();
            }
        } else {
            alert('Please select an image first.');
        }
    });

    scaleUpButton.addEventListener('click', () => {
        if (selectedImage) {
            const scaleFactor = 1.1;
            const newWidth = Math.round(selectedImage.width * scaleFactor);
            const newHeight = Math.round(selectedImage.height * scaleFactor);
            if (newWidth <= 1024 && newHeight <= 1024) {
                selectedImage.width = newWidth;
                selectedImage.height = newHeight;
                drawCanvas();
            } else {
                alert('Cannot scale up: Image would exceed maximum size (1024x1024).');
            }
        } else {
            alert('Please select an image first.');
        }
    });

    scaleDownButton.addEventListener('click', () => {
        if (selectedImage) {
            const scaleFactor = 0.9;
            const newWidth = Math.round(selectedImage.width * scaleFactor);
            const newHeight = Math.round(selectedImage.height * scaleFactor);
            if (newWidth >= 10 && newHeight >= 10) {
                selectedImage.width = newWidth;
                selectedImage.height = newHeight;
                drawCanvas();
            } else {
                alert('Cannot scale down: Image would be too small.');
            }
        } else {
            alert('Please select an image first.');
        }
    });

    flipHorizontalButton.addEventListener('click', () => {
        if (selectedImage) {
            selectedImage.flipX = !selectedImage.flipX;
            drawCanvas();
        } else {
            alert('Please select an image first.');
        }
    });

    processButton.addEventListener('click', async () => {
        if (images.length === 0) {
            alert('No images to process!');
            return;
        }
        try {
            const canvasPngBase64 = canvas.toDataURL('image/png');
            const textboxEl = [...document.querySelectorAll('label')]
                .find(label => label.innerText.includes('bridge'))
                ?.parentElement.querySelector('textarea');
            const btnEl = document.getElementById("button");

            if (textboxEl && btnEl) {
                textboxEl.value = canvasPngBase64;
                textboxEl.dispatchEvent(new Event('input', { bubbles: true }));
                btnEl.click();
            } else {
                alert('Could not find Gradio input components.');
            }
        } catch (error) {
            console.error('Error processing images:', error);
            alert(`Error processing images: ${error.message || error}`);
        }
    });

    function handleExampleImage(base64) {
        const img = new Image();
        img.src = base64;
        img.onload = () => {
            let width = img.width;
            let height = img.height;

            if (width > 512 || height > 512) {
                const aspectRatio = width / height;
                if (width > height) {
                    width = 512;
                    height = Math.round(512 / aspectRatio);
                } else {
                    height = 512;
                    width = Math.round(512 * aspectRatio);
                }
            }
            window.images = window.images || [];
            window.images.push({
                img: img,
                base64: base64,
                x: 10,
                y: 10,
                width: width,
                height: height,
                flipX: false,
            });
            if (window.drawCanvas) window.drawCanvas();
        };
    }

    function imageUrlToBase64(url, callback) {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const base64 = canvas.toDataURL("image/png");
            callback(base64);
        };
        img.onerror = () => {
            console.error("Failed to load image from URL:", url);
            alert("Error loading image");
        };
        img.src = url;
    }

    const exampleInput = document.getElementById("example_bridge");
    exampleInput.addEventListener("click", (event) => {
        event.preventDefault();
        const imgElement = event.target.querySelector("img") || event.target.closest("div").querySelector("img");
        if (imgElement) {
            console.log("Image element found:", imgElement);
            const imageUrl = imgElement.src;
            console.log("Image URL:", imageUrl);
            imageUrlToBase64(imageUrl, (base64) => {
                handleExampleImage(base64);
            });
        } else {
            console.log("No <img> element found");
            alert("No image found in the clicked element");
        }
    });

    exampleButton.addEventListener('click', async () => {
        exampleInput.click();
    });

    window.addEventListener('resize', drawCanvas);
    window.images = images;
    window.drawCanvas = drawCanvas;
}