const teamLinks = document.querySelectorAll('#team-list li');
const userNameDisplay = document.getElementById('user-name');
const embedContainer = document.getElementById('embed-container');

teamLinks.forEach(link => {
  link.addEventListener('click', () => {
    // Update active state
    document.querySelector('li.active').classList.remove('active');
    link.classList.add('active');

    const userKey = link.dataset.user;
    const name = link.textContent;
    userNameDisplay.textContent = name;

    // Dynamic iframe loaders per person
    switch (userKey) {
      case "imran":
        embedContainer.innerHTML = `
          <iframe src="/templates" frameborder="0" width="100%" height="700px"></iframe>
        `;
        break;

      case "chi":

        break;

      case "daryl":

        break;

      case "zach":
 
        break;

      case "vanessa":
      
        break;

      case "simon":
    
        break;

      case "joshua":
   
        break;

      case "leon":

        break;

      default:

    }
  });
});
