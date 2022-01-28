<template>


  <div class="container">
    <div class="row">
      <div class="col-lg-12" style="margin-bottom: 20px">
        <div class="box-main" style="border-color: #3949AB; height: 100%">
          <div class="col-lg-12 box-title" style="background-color: #C5CAE9">
            <h3>EVENT</h3>
          </div>
          <div class="center-element">
            <h1><u>E</u>lectric <u>VE</u>hicle <u>N</u>etwork analysis <u>T</u>ool</h1>
          </div>

          <div class="center-element">
            <!-- <div class="img-fluid"><img src="../assets/images/220127-icons.png" alt="Responsive image">
            </div> -->
           <img src="../assets/images/220127_icons.svg" class="img-fluid" alt="Responsive image" height: auto> 
          </div>
          
            <div class="center-element">
            
            <div class="col-lg-4 offset-4">
            <router-link
              class="btn btn-primary btn-block"
              style="background-color: #3949AB; border-color: #3949AB; font-size: 24px"
              :to="{ name: 'SimulateNetworkAPI' }"
              >Build a Network
            </router-link>
            </div>
          </div>
          <div class="row" style="padding: 60px 20px 20px">
            <img class="col-md-3 logo-main" src="../assets/logos/supergen.png" alt="Responsive image">
            <img class="col-md-3 logo-main" src="../assets/logos/turing.png" alt="Responsive image">
            <img class="col-md-3 logo-main" src="../assets/logos/newcastle.png" alt="Responsive image">
            <img class="col-md-3 logo-main" src="../assets/logos/lrf.svg" alt="Responsive image">
          </div>

        <!-- </div>
      </div>
    </div>
    <div class="row">
      <div class="col-lg-12" style="margin-bottom: 20px">
        <div class="box-main" style="border-color: #1E88E5; height: 100%">
          <div class="col-lg-12 box-title" style="background-color: #BBDEFB">
            <h3>Acknowledgements</h3>
          </div>
          <div class="col-lg-12">
            <p>This work is a collaboration of Newcastle University, Research Engineering at The Alan Turing Institute and the Supergen Energy Networks Hub.</p>
            <p>For more information see <a href="https://www.turing.ac.uk/research/research-projects/vehicle-grid-integration">the project website</a>.</p>
            <p>This work is funded by the e4Future grant (104227) - under UK Research and Innovation by the Department for Business, Energy and Industrial Strategy (BEIS) and The Office for Zero Emission Vehicles (OZEV) and delivered through Innovate UK; and a grant from the Lloyd's Register Foundation (G0095) under the Data-Centric Engineering Programme of The Alan Turing Institute.</p>
          </div> -->
        </div>
      </div>
    </div>
  </div>

<footer class="footer mt-auto py-3">
  <div class="container text-center">
    <span class="text-muted">Copyright &copy; 2019-2022 <strong>e4futuregrid.com</strong>. All rights reserved.</span>
  </div>
</footer>

</template>
