const form = document.getElementById('predict-form');
const predEl = document.getElementById('prediction');
const clearBtn = document.getElementById('clear-btn');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  predEl.textContent = '...';

  const body = {
    sepal_length: parseFloat(document.getElementById('sepal_length').value),
    sepal_width: parseFloat(document.getElementById('sepal_width').value),
    petal_length: parseFloat(document.getElementById('petal_length').value),
    petal_width: parseFloat(document.getElementById('petal_width').value)
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (res.ok) {
      predEl.textContent = String(data.response);
    } else {
      predEl.textContent = 'error';
      console.error(data);
    }
  } catch (err) {
    predEl.textContent = 'error';
    console.error(err);
  }
});

clearBtn.addEventListener('click', () => {
  document.getElementById('sepal_length').value = '';
  document.getElementById('sepal_width').value = '';
  document.getElementById('petal_length').value = '';
  document.getElementById('petal_width').value = '';
  predEl.textContent = '—';
});
