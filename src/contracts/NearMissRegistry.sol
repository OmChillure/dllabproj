// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract NearMissRegistry {
    struct Incident {
        uint256 id;
        uint64 occurredAt;
        bytes32 cameraIdHash;
        string vehicleClass;
        uint32 distanceCm;
        uint32 ttcMs;
        uint8 severityScore;
        string severityLabel;
        bool alertFlag;
        string clipCid;
        string snapshotCid;
        address reporter;
    }

    uint256 public nextIncidentId = 1;
    mapping(uint256 => Incident) private incidents;

    event IncidentRecorded(
        uint256 indexed incidentId,
        uint64 occurredAt,
        bytes32 indexed cameraIdHash,
        string vehicleClass,
        uint8 severityScore,
        bool alertFlag,
        string clipCid,
        string snapshotCid,
        address indexed reporter
    );

    function recordIncident(
        uint64 occurredAt,
        bytes32 cameraIdHash,
        string calldata vehicleClass,
        uint32 distanceCm,
        uint32 ttcMs,
        uint8 severityScore,
        string calldata severityLabel,
        bool alertFlag,
        string calldata clipCid,
        string calldata snapshotCid
    ) external returns (uint256 incidentId) {
        require(occurredAt > 0, "invalid timestamp");
        require(bytes(vehicleClass).length > 0, "vehicle class required");
        require(bytes(clipCid).length > 0, "clip CID required");
        require(severityScore <= 100, "score out of range");

        incidentId = nextIncidentId++;

        incidents[incidentId] = Incident({
            id: incidentId,
            occurredAt: occurredAt,
            cameraIdHash: cameraIdHash,
            vehicleClass: vehicleClass,
            distanceCm: distanceCm,
            ttcMs: ttcMs,
            severityScore: severityScore,
            severityLabel: severityLabel,
            alertFlag: alertFlag,
            clipCid: clipCid,
            snapshotCid: snapshotCid,
            reporter: msg.sender
        });

        emit IncidentRecorded(
            incidentId,
            occurredAt,
            cameraIdHash,
            vehicleClass,
            severityScore,
            alertFlag,
            clipCid,
            snapshotCid,
            msg.sender
        );
    }

    function getIncident(uint256 incidentId) external view returns (Incident memory) {
        require(exists(incidentId), "incident not found");
        return incidents[incidentId];
    }

    function exists(uint256 incidentId) public view returns (bool) {
        return incidents[incidentId].reporter != address(0);
    }
}

