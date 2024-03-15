document.getElementById('btn-submit').addEventListener('click', (event) => {
    event.preventDefault()
    const selectedModel = document.getElementById('modelDropdown').value;
    
    fetch(`http://localhost:8000/choose_model?model_name=${selectedModel}`)
    .then(response => response.blob())
    .then(blob => {
        const imgUrl = URL.createObjectURL(blob);
        document.getElementById('result_plot').innerHTML = `<img src="${imgUrl}" alt="Plot Image">`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});