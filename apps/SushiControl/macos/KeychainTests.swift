import Foundation

@main struct KeychainTests {
    static func main() throws {
        let id = "test-" + UUID().uuidString
        let vault = CredentialStore()
        defer { try? vault.remove(id) }
        try vault.write("test-only-first",id:id)
        precondition(vault.read(id) == "test-only-first")
        try vault.write("test-only-updated",id:id)
        precondition(vault.read(id) == "test-only-updated")
        try vault.remove(id)
        precondition(vault.read(id) == nil)
        print("Keychain checks passed: create, read, update, remove")
    }
}
