// script.js

document.addEventListener('DOMContentLoaded', () => {
    const openBtns = document.querySelectorAll('.open-btn');

    openBtns.forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = btn.getAttribute('data-section-id');
            const filesContainer = document.getElementById(sectionId + '-files');

            // Load and display attached files
            // Replace this with your actual file loading logic
            const files = [
                { name: 'File 1', url: 'file1.txt' },
                { name: 'File 2', url: 'file2.pdf' },
                { name: 'File 3', url: 'file3.docx' },
            ];

            files.forEach((file) => {
                const fileLink = document.createElement('a');
                fileLink.href = file.url;
                fileLink.download = file.name;
                fileLink.click();

                // Create a link to view the file content
                const viewLink = document.createElement('a');
                viewLink.href = file.url;
                viewLink.target = '_blank';
                viewLink.textContent = 'View File';

                filesContainer.appendChild(viewLink);
            });
        });
    });
});
