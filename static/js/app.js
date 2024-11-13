document.addEventListener('DOMContentLoaded', () => {
     // Example: Form validation feedback
     const form = document.querySelector('form');
     form.addEventListener('submit', (event) => {
         const fileInput = form.querySelector('input[type="file"]');
         if (!fileInput.files.length) {
             alert('Please upload a document.');
             event.preventDefault();
         }
     });
 });
 