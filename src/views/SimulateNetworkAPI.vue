<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-12">
        <h3>Build and simulate an electricity distribution network API</h3>
        <p>
          You can find more information on OpenDSS parameters below in the
          <a
            href="http://svn.code.sf.net/p/electricdss/code/trunk/Distrib/Doc/OpenDSSManual.pdf"
            >OpenDSS Manual</a
          >.
        </p>
      </div>
    </div>

    <form id="config" ref="config" @submit.prevent>
      <div class="row">
        <div class="col-lg-12">
          <h4>Electricity distribution network parameters</h4>
        </div>

        <div class="col-lg-6">
          <h5>Medium voltage (MV)</h5>

          <div class="form-group row">
            <!-- Experiment parameter: n_id, network ID -->
            <label for="n_id" class="col-sm-6 col-form-label">
              Network ID
            </label>
            <div class="col-sm-6">
              <select v-model="config.n_id" class="form-control">
                <option value="1060">11kV urban network</option>
                <option value="1061">11kV urban - rural network</option>
              </select>
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: xfmr_scale, MV transformer scaling -->
            <label for="xfmr_scale" class="col-md-6 col-form-label">
              MV transformer scaling
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.xfmr_scale"
                type="float"
                class="form-control"
                id="xfmr_scale"
                placeholder="xfmr_scale"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: oltc_setpoint -->
            <label for="oltc_setpoint" class="col-md-6 col-form-label">
              MV transformer on-load tap charger (OLTC) set point
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.oltc_setpoint"
                type="float"
                class="form-control"
                id="oltc_setpoint"
                placeholder="OLTC set point, e.g. 1.04"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: oltc_bandwidth -->
            <label for="oltc_bandwidth" class="col-md-6 col-form-label">
              MV transformer on-load tap charger (OLTC) bandwidth
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.oltc_bandwidth"
                type="float"
                class="form-control"
                id="oltc_bandwidth"
                placeholder="OLTC bandwidth, e.g. 0.13"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: rs_pen -->
            <label for="rs_pen" class="col-md-6 col-form-label">
              Percentage residential loads
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.rs_pen"
                type="float"
                class="form-control"
                id="rs_pen"
                placeholder="Percentage residential loads e.g. 0.8"
              />
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <h5>Low voltage (LV)</h5>

          <div class="form-group row">
            <!-- Experiment parameter: lv_default (if custom, open lv_list option below) -->
            <label for="lv_default" class="col-md-6 col-form-label">
              IDs of up to 5 LV networks to model in detail
            </label>
            <div class="col-md-6">
              <select v-model="config.lv_default" class="form-control">
                <option>1101, 1105, 1103</option>
                <option>1101, 1102, 1103</option>
                <option>Custom</option>
              </select>
            </div>
          </div>

          <template v-if="config.lv_default == 'Custom'">
            <div class="form-group row">
              <!-- Experiment parameter: lv_list -->
              <label for="lv_list" class="col-md-6 col-form-label">
                Custom selection of up to 5 LV network IDs to model in detail
              </label>
              <div class="col-md-6">
                <input
                  v-model.number="config.lv_list"
                  type="string"
                  class="form-control"
                  id="lv_list"
                  placeholder="1101, 1105, 1103"
                />
              </div>
            </div>
          </template>
        </div>
      </div>

      <div class="row">
        <div class="col-lg-12">
          <h4>Demand and generation profiles</h4>
        </div>

        <div class="col-lg-6">
          <h5>MV connected</h5>

          <div class="form-group row">
            <!-- Experiment parameter: mv_solar_pv_profile (if custom, open upload csv option below) -->
            <label for="mv_solar_pv_profile" class="col-md-6 col-form-label">
              11kV connected solar PV profile
            </label>
            <div class="col-md-6">
              <select v-model="config.mv_solar_pv_profile" class="form-control">
                <option
                  v-for="opt in profile_options.mv_solar_pv_list"
                  :key="opt"
                  >{{ opt }}</option
                >
              </select>
            </div>
          </div>

          <template v-if="config.mv_solar_pv_profile == 'csv'">
            <!-- Upload: mv_solar_pv_csv -->
            <div class="form-group row">
              <label for="mv_solar_pv_csv" class="col-md-6 col-form-label">
                Custom 11kV connected solar PV profile
              </label>
              <div class="col-md-6">
                <input
                  type="file"
                  class="form-control"
                  id="mv_solar_pv_csv"
                  placeholder="mv_solar_pv_csv"
                />
              </div>
            </div>
            <div class="form-group row">
              <label class="col-md-6 col-form-label">Units</label>
              <div class="col-md-6">
                <div class="form-check form-check-inline">
                  <input
                    v-model="config.mv_solar_pv_profile_units"
                    type="radio"
                    class="form-check-input"
                    name="pv_unit_kW"
                    value="kW"
                  />
                  <label class="form-check-label" for="pv_unit_kW">kW</label>
                </div>
                <div class="form-check form-check-inline">
                  <input
                    v-model="config.mv_solar_pv_profile_units"
                    type="radio"
                    class="form-check-input"
                    name="pv_unit_kWh"
                    value="kWh"
                  />
                  <label class="form-check-label" for="pv_unit_kWh">kWh</label>
                </div>
              </div>
            </div>
          </template>

          <div class="form-group row">
            <!-- Experiment parameter: mv_ev_charger_profile (if custom, open upload csv option below) -->
            <label for="mv_ev_charger_profile" class="col-md-6 col-form-label">
              11kV connected electric vehicle charging profile
            </label>
            <div class="col-md-6">
              <select
                v-model="config.mv_ev_charger_profile"
                class="form-control"
              >
                <option
                  v-for="opt in profile_options.mv_ev_charger_list"
                  :key="opt"
                  >{{ opt }}</option
                >
              </select>
            </div>
          </div>

          <template v-if="config.mv_ev_charger_profile == 'csv'">
            <!-- Upload: mv_ev_charger_csv -->
            <div class="form-group row">
              <label for="mv_ev_charger_csv" class="col-md-6 col-form-label">
                Custom 11kV connected electric vehicle charging profile
              </label>
              <div class="col-md-6">
                <input
                  type="file"
                  class="form-control"
                  id="mv_ev_charger_csv"
                  placeholder="mv_ev_charger_csv"
                />
              </div>
            </div>
            <div class="form-group row">
              <label class="col-md-6 col-form-label">Units</label>
              <div class="col-md-6">
                <div class="form-check form-check-inline">
                  <input
                    v-model="config.mv_ev_charger_profile_units"
                    type="radio"
                    class="form-check-input"
                    name="ev_unit_kW"
                    value="kW"
                  />
                  <label class="form-check-label" for="ev_unit_kW">kW</label>
                </div>
                <div class="form-check form-check-inline">
                  <input
                    v-model="config.mv_ev_charger_profile_units"
                    type="radio"
                    class="form-check-input"
                    name="ev_unit_kWh"
                    value="kWh"
                  />
                  <label class="form-check-label" for="ev_unit_kWh">kWh</label>
                </div>
              </div>
            </div>
          </template>
        </div>

        <div class="col-lg-6">
          <h5>LV connected</h5>
        </div>
      </div>

      <div class="row">
        <div class="col-lg-12">
          <h5>Run simulation</h5>
          <button type="submit" class="btn btn-primary" @click="fetchAPIData">
            Submit
            <template v-if="isLoading">
              <div class="spinner-border spinner-border-sm" role="status"></div>
            </template>
          </button>
        </div>
      </div>

      <div class="row">
        <div class="col-lg-12">
          <h5>Results</h5>
        </div>
      </div>
    </form>

    <template v-if="responseAvailable">
      <div class="accordion" id="accordionResults">
        <div class="card">
          <div class="card-header">
            <h2 class="mb-0">
              <button
                class="btn btn-link btn-block text-left"
                type="button"
                data-toggle="collapse"
                data-target="#jsonCollapse"
              >
                DEBUG: Show JSON response
              </button>
            </h2>
          </div>
          <div
            id="jsonCollapse"
            class="collapse"
            data-parent="#accordionResults"
          >
            <div class="card-body">
              {{ rawJson }}
            </div>
          </div>
        </div>

        <div v-for="(p, ind) in plots" :key="p.ind">
          <div class="card">
            <div class="card-header">
              <h2 class="mb-0">
                <button
                  class="btn btn-link btn-block text-left"
                  type="button"
                  data-toggle="collapse"
                  :data-target="'#plotCollapse' + ind"
                >
                  {{ p.name }}
                </button>
              </h2>
            </div>
            <div
              :id="'plotCollapse' + ind"
              class="collapse"
              v-bind:class="{ show: !ind }"
              data-parent="#accordionResults"
            >
              <div class="card-body">
                <img v-bind:src="'data:image/jpeg;base64,' + p.plot" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script>
export default {
  el: "#main",

  data() {
    return {
      config: {
        // Electricity distribution network parameters
        // MV
        n_id: 1060,
        xfmr_scale: 1.0,
        oltc_setpoint: 1.04,
        oltc_bandwidth: 0.13,
        rs_pen: 0.8,
        // LV
        lv_default: "1101, 1105, 1103",
        lv_list: "1101, 1105, 1103",
        // Demand and generation profiles
        // MV
        mv_solar_pv_profile: "Option a",
        mv_solar_pv_profile_units: "kW",
        mv_solar_pv_csv: null,
        mv_ev_charger_profile: "Option d",
        mv_ev_charger_profile_units: "kW",
        mv_ev_charger_csv: null,
        // LV
        p_ev: 10
      },
      profile_options: {
        mv_solar_pv_list: [],
        mv_ev_charger_list: []
      },
      // voltages: [],
      // report: [],
      plot1: "",
      plot2: "",
      rawJson: "empty",
      isShowJson: false,
      isLoading: false,
      responseAvailable: false
    };
  },

  mounted() {
    // Populate the lists used in the dropdown menus with their options
    this.getProfileOptions("mv-solar-pv").then(p_list => {
      this.profile_options.mv_solar_pv_list = p_list;
      this.config.mv_solar_pv_profile = p_list[0];
    });
    this.getProfileOptions("mv-ev-charger").then(p_list => {
      this.profile_options.mv_ev_charger_list = p_list;
      this.config.mv_ev_charger_profile = p_list[0];
    });
  },

  methods: {
    fetchAPIData() {
      // Ask user to wait while API request is formed and made
      this.plot1 = "";
      this.plot2 = "";
      this.plots = [];
      this.rawJson = "wait...";
      this.isLoading = true;
      this.responseAvailable = false;

      var url = process.env.VUE_APP_API_URL + "/simulate";

      if (process.env.NODE_ENV == "development") {
        console.log("Using API URL:", url);
      }

      // Convert config options into URL parameters

      // Electricity distribution network parameters / MV
      let edn_params_mv = "n_id=" + this.config.n_id;
      "&xfmr_scale=" +
        this.config.xfmr_scale +
        "&oltc_setpoint=" +
        this.config.oltc_setpoint +
        "&oltc_bandwidth=" +
        this.config.oltc_bandwidth +
        "&rs_pen=" +
        this.config.rs_pen;

      // Electricity distribution network parameters / LV
      let edn_params_lv = "lv_list=" + this.config.lv_list;

      // Demand and generation profiles / MV
      let dag_params_mv = "";

      url +=
        "?" +
        edn_params_mv +
        "&" +
        edn_params_lv +
        dag_params_mv +
        "&dry_run=true";
      console.log("url: ", url);

      // let formData = new FormData();
      fetch(url, {
        method: "POST"
        // body: formData  // don't try to upload any csv data for now
      })
        .then(response => {
          if (response.ok) {
            return response.text();
          } else {
            alert(
              "Server returned " + response.status + " : " + response.statusText
            );
          }
        })
        .then(response => {
          // Populate rawJson
          this.rawJson = response;

          // Parse json repsonse
          var responseJson = JSON.parse(response);

          // // Parse revoltages port from json to array to list
          // this.voltages = responseJson["voltages"];

          // // Parse report from json to array to list
          // this.report = responseJson["report"];

          // Parse plot from json to image data
          this.plot1 = responseJson["plot1"];
          this.plot2 = responseJson["plot2"];
          this.plots = [
            { name: "Network layout", plot: responseJson["plot1"] },
            { name: "MV load over time", plot: responseJson["plot2"] }
          ];

          this.responseAvailable = true;
          this.isLoading = false;
        })
        .catch(err => {
          console.log(err);
        });
    },

    getProfileOptions(profile_name) {
      return fetch(
        process.env.VUE_APP_API_URL +
          "/get-options?option_type=" +
          profile_name,
        {
          method: "GET"
        }
      )
        .then(response => {
          if (response.ok) {
            return response.text();
          } else {
            alert(
              "Server returned " + response.status + " : " + response.statusText
            );
          }
        })
        .then(response => {
          console.log("Returning response ", response);
          return JSON.parse(response);
        })
        .catch(err => {
          console.log(err);
        });
    }
  }
};
</script>

<style scoped>
.form-check {
  margin-bottom: 15px;
}
#displayBox {
  margin-top: 20px;
}
</style>
