document.addEventListener('DOMContentLoaded', () => {
  const files = {
    chi: [{ name: 'Chi - Code Explanation', url: 'Chi.pdf' }],
    daryl: [{ name: 'Daryl - Code Explanation', url: 'Daryl.pdf' }],
    imran: [{ name: 'Imran - Code Explanation', url: 'Imran.pdf' }],
    zach: [{ name: 'Zach - Code Explanation', url: 'Zach.pdf' }],
    venessa: [{ name: 'Venessa - Code Explanation', url: 'Vanessa.pdf' }],
    simon: [{ name: 'Simon - Code Explanation', url: 'Simon.pdf' }],
    joshua: [{ name: 'Joshua - Code Explanation', url: 'Joshua_Vireak.pdf' }],
    leon: [{ name: 'Leon - Code Explanation', url: 'Leon.pdf' }]
  };

  document.querySelectorAll('.open-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sectionId = btn.getAttribute('data-section-id');
      const filesContainer = document.getElementById(`${sectionId}-files`);
      filesContainer.innerHTML = ''; // Clear previous content

      files[sectionId].forEach(file => {
        const link = document.createElement('a');
        link.href = file.url;
        link.target = '_blank';
        link.textContent = `📄 ${file.name}`;
        filesContainer.appendChild(link);
      });
    });
  });
});
