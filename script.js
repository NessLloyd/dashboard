document.addEventListener('DOMContentLoaded', () => {
  const files = {
    chi: [{ name: 'Chi's Code', url: 'Chi.pdf' }],
    daryl: [{ name: 'Daryl's Code', url: 'Daryl.pdf' }],
    imran: [{ name: 'Imran's Code', url: 'Imran.pdf' }],
    zach: [{ name: 'Zach's Code', url: 'Zach.pdf' }],
    vanessa: [{ name: 'Vanessa's Code', url: 'Vanessa.pdf' }],
    simon: [{ name: 'Simon's Code', url: 'Simon.pdf' }],
    joshua: [{ name: 'Joshua and Vireak's Code', url: 'Joshua_Vireak.pdf' }],
    leon: [{ name: 'Leon's Code', url: 'Leon.pdf' }]
  };

  document.querySelectorAll('.open-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      const sectionId = btn.getAttribute('data-section-id');
      const filesContainer = document.getElementById(`${sectionId}-files`);
      filesContainer.innerHTML = ''; // clear if already loaded

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

