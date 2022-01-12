<template>

  <div class="container">
    <div class="row">
      <div class="col-lg-6" style="margin-bottom: 20px">
        <div class="box-main" style="border-color: #3949AB; height: 100%">
          <div class="col-lg-12 box-title" style="background-color: #C5CAE9">
            <h3>EVENT</h3>
          </div>
          <div class="center-element">
            <h5><u>E</u>lectric <u>VE</u>hicle <u>N</u>etwork analysis <u>T</u>ool</h5>
          </div>
          <div class="center-element">
            <router-link
              class="btn btn-primary"
              style="background-color: #3949AB; border-color: #3949AB"
              :to="{ name: 'SimulateNetworkAPI' }"
              >Build a Network
            </router-link>
          </div>
        </div>
      </div>

      <div class="col-lg-6" style="margin-bottom: 20px">
        <div class="box-main" style="border-color: #1E88E5; height: 100%">
          <div class="col-lg-12 box-title" style="background-color: #BBDEFB">
            <h3>Acknowledgements</h3>
          </div>
          <div class="col-lg-12">
            <p>This work is a collaboration of Newcastle University, Research Engineering at The Alan Turing Institute and the Supergen Energy Networks Hub.</p>
            <p>For more information see <a href="https://www.turing.ac.uk/research/research-projects/vehicle-grid-integration">the project website</a>.</p>
            <p>This work is funded by the e4Future grant (104227) - under UK Research and Innovation by the Department for Business, Energy and Industrial Strategy (BEIS) and The Office for Zero Emission Vehicles (OZEV) and delivered through Innovate UK; and a grant from the Lloyd's Register Foundation (G0095) under the Data-Centric Engineering Programme of The Alan Turing Institute.</p>
          </div>
          <div class="row" style="padding: 20px 15px 10px">
            <img class="col-md-4 logo" src="../assets/logos/supergen.png">
            <img class="col-md-4 logo" src="../assets/logos/turing.png">
            <img class="col-md-4 logo" src="../assets/logos/newcastle.png">
          </div>
        </div>
      </div>
    </div>
  </div>

</template>
