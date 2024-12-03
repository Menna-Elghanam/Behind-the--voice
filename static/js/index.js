var toggler =true;

function toggle(){
    if(toggler){
        document.getElementById("nav-toggler").style.backgroundImage= `url()`;
        document.getElementById("nav-toggler").innerHTML='<i class="fa-solid fa-xmark"></i>';
        toggler = false;
    }
    else{
        document.getElementById("nav-toggler").style.backgroundImage= `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'%3e%3cpath stroke='rgba%280, 0, 0, 0.55%29' stroke-linecap='round' stroke-miterlimit='10' stroke-width='2' d='M4 7h22M4 15h22M4 23h22'/%3e%3c/svg%3e")`;
        document.getElementById("nav-toggler").innerHTML=' ';
        toggler = true;

    }
}




var ContactUsFirstName = document.getElementById("ContactUsFirstName");
var ContactUsLastName = document.getElementById("ContactUsLastName");
var ContactUsEmail = document.getElementById("ContactUsEmail");
var ContactUsPhone = document.getElementById("ContactUsPhone");
function validateContactUsFirstName() {
  var regex = /^[a-zA-Z]{2,10}$/;

  if (regex.test(ContactUsFirstName.value)) {
    ContactUsFirstName.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    ContactUsFirstName.classList.add("is-invalid");
    return false;
  }
}
function validatevalidateContactUsLastName() {
  var regex = /^[a-zA-Z]{2,10}$/;
  if (regex.test(ContactUsLastName.value)) {
    ContactUsLastName.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    ContactUsLastName.classList.add("is-invalid");
    return false;
  }
}
function validateContactUsEmail() {
  var regex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

  if (regex.test(ContactUsEmail.value)) {
    ContactUsEmail.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    ContactUsEmail.classList.add("is-invalid");
    return false;
  }
}
function validateContactUsPhone() {
  var regex = /^01[0-25][0-9]{8}$/;
  if (regex.test(ContactUsPhone.value)) {
    ContactUsPhone.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    ContactUsPhone.classList.add("is-invalid");
    return false;
  }
}
function validateContactUsData() {
  if (
    validateContactUsFirstName() &&
    validatevalidateContactUsLastName() &&
    validateContactUsEmail() &&
    validateContactUsPhone() == true
  ) {
    return true;
  } else {
    return false;
  }
}


/* try animation */
const animatedSection = document.querySelector('.feature');

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      animatedSection.classList.add('in-view');
    } 
    else {
      animatedSection.classList.remove('in-view');
    }
  });
});
observer.observe(animatedSection);



const animatedSection2 = document.querySelector('.feature_2');

const observer2 = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      animatedSection2.classList.add('in-view');
    } 
    else {
      animatedSection2.classList.remove('in-view');
    }
  });
});

observer2.observe(animatedSection2);




const animatedSection3 = document.querySelector('.feature_3');

const observer3 = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      animatedSection3.classList.add('in-view');
    } 
    else {
      animatedSection3.classList.remove('in-view');
    }
  });
});

observer3.observe(animatedSection3);

