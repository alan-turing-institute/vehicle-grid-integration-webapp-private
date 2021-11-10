<template>
  <div class="row">
    <div class="card mr-2" style="width: 20rem;">
      <img
        src="https://images.unsplash.com/photo-1454779132693-e5cd0a216ed3?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2251&q=80"
        class="card-img-top"
        alt="unsplash-image"
      />
      <div class="card-body">
        <h5 class="card-title">Electricity distribution network</h5>
        <p class="card-text">
          Build and simulate an electricity distribution network, using OpenDSS.
        </p>
        <!--
        <router-link class="btn btn-primary" :to="{ name: 'SimulateNetwork' }"
          >Build a Network</router-link
        >
        -->
        <p>Deprecated</p>
      </div>
    </div>
    <div class="card" style="width: 20rem;">
      <img
        src="https://images.unsplash.com/photo-1454779132693-e5cd0a216ed3?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2251&q=80"
        class="card-img-top"
        alt="unsplash-image"
      />
      <div class="card-body">
        <h5 class="card-title">Electricity distribution network API</h5>
        <p class="card-text">
          Build and simulate an electricity distribution network, using OpenDSS.
        </p>
        <router-link
          class="btn btn-primary"
          :to="{ name: 'SimulateNetworkAPI' }"
          >Build a Network</router-link
        >
      </div>
    </div>
  </div>
</template>
